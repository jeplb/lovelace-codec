"""original ordinary run profile parser for orientation, primer and cell extraction."""

import argparse
import base64
from collections import Counter
import ctypes as ct
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

import numpy as np

BASES = "ACT"
DATA_PER_IMAGE = 224
DATA_OLIGOS = 448
TOTAL_OLIGOS = 480
ADDRESS_RUNS = 16
ADDRESS_CAPACITY = 2048
MASK_BYTES = 256
STATES = np.array(((3, 7, 10), (3, 10, 7), (7, 3, 10), (7, 10, 3)), np.int32)
CONFIG = {
    "address_margin": 1.5,
    "byte_confidence": 0.75,
    "coefficient_confidence": 0.75,
    "max_primer_edits": 6,
    "minimum_address_score": -85.0,
    "minimum_payload_score": -125.0,
    "minimum_pilot_row_count": 8,
    "prior_shrinkage": 16.0,
}
PREFIX = struct.Struct(">4sBBBBBBHddI")
PARSER_SOURCE = r'''
// viterbi run profile. maxima give hard alignments, no marginalization
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;
struct Run { char b; int n; };
static vector<Run> rle(const string&s){vector<Run> r;for(char c:s){if(!r.empty()&&r.back().b==c)r.back().n++;else r.push_back({c,1});}return r;}
static string collapse(string s){string r;for(char c:s)if(r.empty()||r.back()!=c)r+=c;return r;}
static int bi(char c){return c=='A'?0:c=='C'?1:c=='T'?2:3;}
static string revcomp(string s){reverse(s.begin(),s.end());for(char&c:s)c=c=='A'?'T':c=='T'?'A':c=='C'?'G':'C';return s;}
static const string p1="CCCAATTAATACGTTGCCTACTCCG",p2="CGCTATCAGAACAAGAGAAGCACTA";
static vector<string> addresses;
static unordered_map<string,vector<int>> lookup;
static double logw[3][13][21];
static int states[4][3];
#ifndef POOLED_LOOKUP_CAPACITY
#define POOLED_LOOKUP_CAPACITY 2048
#endif
static constexpr int ADDRESS_RUNS=16, ADDRESS_CAPACITY=POOLED_LOOKUP_CAPACITY, MASK_BYTES=256, ALLOCATED=480;
static void add(const string&s,int id){auto &v=lookup[collapse(s)];if(v.empty()||v.back()!=id)v.push_back(id);}
extern "C" void init_parser(const char*patterns,const double*w,const int*ss){
  addresses.clear();lookup.clear();
  for(int b=0;b<3;b++)for(int x=0;x<13;x++)for(int y=0;y<21;y++)logw[b][x][y]=log(max(w[(b*13+x)*21+y],1e-12));
  for(int s=0;s<4;s++)for(int b=0;b<3;b++)states[s][b]=ss[s*3+b];
  for(int id=0;id<ADDRESS_CAPACITY;id++){
    string s(patterns+id*ADDRESS_RUNS,ADDRESS_RUNS); addresses.push_back(s); add(s,id);
    for(int i=0;i<ADDRESS_RUNS;i++){
      string a=s;a.erase(i,1);add(a,id);
      for(int j=i+1;j<ADDRESS_RUNS;j++){string t=s;t.erase(j,1);t.erase(i,1);add(t,id);}
      for(char b:string("ACT")){string t=s;t[i]=b;add(t,id);}
    }
    for(int i=0;i<=ADDRESS_RUNS;i++)for(char b:string("ACT")){string t=s;t.insert(i,1,b);add(t,id);}
  }
}
// semiglobal primer fit, up to four leading contaminant bases
// endpoint near the start. don't invent short truncated primers
// reject damage beyond the edit limit
static pair<int,int> primer(const string&seq,const string&p){
  int n=min((int)seq.size(),48);int prev[49],cur[49];
  for(int j=0;j<=n;j++)prev[j]=max(0,j-4);
  for(int i=1;i<=(int)p.size();i++){
    cur[0]=i;
    for(int j=1;j<=n;j++)cur[j]=min({prev[j]+1,cur[j-1]+1,prev[j-1]+(p[i-1]!=seq[j-1])});
    copy(cur,cur+n+1,prev);
  }
  pair<int,int> best={1000,0};for(int j=15;j<=min(n,40);j++)if(prev[j]<best.first)best={prev[j],j};
  return best;
}
static constexpr int M=96;
struct Step {int8_t di,dj,u;};
// known-base profile. run distributions already include observed zeros
// insertion, split-run and merged-run paths are separately reachable
// traceback gives inferred observations, never ground truth
static double profile(const vector<Run>&obs,const string&pattern,const vector<array<double,21>>&em,
                      int16_t*out,int*events){
  int n=pattern.size(),m=obs.size();if(m>=M||n>32)return -1e90;
  double dp[33][M];Step back[33][M];
  for(int i=0;i<=n;i++)for(int j=0;j<=m;j++){dp[i][j]=-1e90;back[i][j]={0,0,0};}
  dp[0][0]=0;
  auto update=[&](int i,int j,double v,int di,int dj,int u){if(v>dp[i][j]){dp[i][j]=v;back[i][j]={(int8_t)di,(int8_t)dj,(int8_t)u};}};
  for(int i=0;i<=n;i++)for(int j=0;j<=m;j++){
    double v=dp[i][j];if(v<-1e80)continue;
    if(j<m)update(i,j+1,v-6.0-0.35*obs[j].n,0,1,0);
    if(i==n)continue;
    update(i+1,j,v+em[i][0],1,0,0);
    if(j==m)continue;
    int y=obs[j].n;
    if(y<=20)update(i+1,j+1,v+em[i][y]-(obs[j].b==pattern[i]?0:8.0),1,1,y);
    // wrong-base insertion or substitution inside a run splits it
    // combine the same-base fragments and charge a disruption
    if(j+2<m&&obs[j].b==pattern[i]&&obs[j+2].b==pattern[i]&&obs[j+1].n<=2){
      int sum=obs[j].n+obs[j+2].n;
      if(sum<=20)update(i+1,j+3,v+em[i][sum]-5.0-0.35*obs[j+1].n,1,3,sum);
    }
    // missing middle runs can merge two same-base runs
    for(int k=2;k<=3;k++)if(i+k<n&&pattern[i]==pattern[i+k]&&obs[j].b==pattern[i]&&y<=40){
      double deletions=0;for(int h=1;h<k;h++)deletions+=em[i+h][0];
      for(int u=max(1,y-20);u<=min(20,y-1);u++)
        update(i+k+1,j+1,v+deletions+em[i][u]+em[i+k][y-u],k+1,1,u);
    }
  }
  if(out){
    fill(out,out+n,-1);int i=n,j=m;int ins=0,del=0,split=0,merge=0;
    while(i||j){Step s=back[i][j];if(!s.di&&!s.dj)break;
      if(s.di==0)ins++;
      else if(s.di==1){out[i-1]=s.u;if(s.dj==0)del++;if(s.dj==3)split++;}
      else{int begin=i-s.di;out[begin]=s.u;out[i-1]=obs[j-1].n-s.u;
        for(int h=begin+1;h<i-1;h++){out[h]=0;del++;}merge++;}
      i-=s.di;j-=s.dj;
    }
    if(events){events[0]=ins;events[1]=del;events[2]=split;events[3]=merge;}
  }
  return dp[n][m];
}
static vector<array<double,21>> address_emission(const string&p){
  vector<array<double,21>> e(ADDRESS_RUNS);
  for(int i=0;i<ADDRESS_RUNS;i++)for(int y=0;y<=20;y++)e[i][y]=logw[bi(p[i])][2][y];
  return e;
}
extern "C" int parse_structure(const char*raw,int*identity,int*orientation,int*shortlist_size,
                                 double*margin,char*payload,int*payload_size,int16_t*address_y,uint8_t*shortlist_mask,
                                 int max_primer_edits,double address_margin,double minimum_address_score){
  memset(shortlist_mask,0,MASK_BYTES);
  string original(raw),seq;int left=0,right=0;double orientBest=1e9,orientSecond=1e9;int orient=-1;
  if(original.size()<80||original.size()>700)return 1;
  for(int o=0;o<2;o++){
    string s=o?revcomp(original):original;
    auto a=primer(s,p1);string rev=s;reverse(rev.begin(),rev.end());string rp=p2;reverse(rp.begin(),rp.end());auto z=primer(rev,rp);
    double score=a.first+z.first;
    if(a.first<=max_primer_edits&&z.first<=max_primer_edits&&score<orientBest){orientSecond=orientBest;orientBest=score;orient=o;seq=s;left=a.second;right=s.size()-z.second;}
    else orientSecond=min(orientSecond,score);
  }
  if(orient<0||orientSecond<=orientBest)return 2;
  if(right-left<40||right-left>350)return 3;
  string body=seq.substr(left,right-left);auto rr=rle(body);
  vector<int> anchors;
  int pos=0;vector<int> starts;
  for(int j=0;j<(int)rr.size();j++){starts.push_back(pos);pos+=rr[j].n;if(rr[j].b=='G'&&rr[j].n>=2)anchors.push_back(j);}
  double best=-1e90,second=-1e90;int bestid=-1,ba=-1,bz=-1;int16_t bestY[ADDRESS_RUNS];int totalCandidates=0;
  for(int z:anchors){if(pos-starts[z]-rr[z].n>6)continue;
    for(int a:anchors){if(a>=z||starts[a]<20||starts[a]>170)continue;
      int plen=starts[z]-starts[a]-rr[a].n;if(plen<25||plen>240)continue;
      vector<Run> addr(rr.begin(),rr.begin()+a);string pattern;for(auto r:addr)pattern+=r.b;
      vector<int> candidates;auto collect=[&](const string&key){auto it=lookup.find(key);if(it!=lookup.end())candidates.insert(candidates.end(),it->second.begin(),it->second.end());};
      collect(pattern);
      if(candidates.empty())for(int k=0;k<(int)addr.size();k++)if(addr[k].n<=2){string t=pattern;t.erase(k,1);collect(collapse(t));}
      sort(candidates.begin(),candidates.end());candidates.erase(unique(candidates.begin(),candidates.end()),candidates.end());
      totalCandidates+=candidates.size();if(candidates.size()>96)continue;
      for(int id:candidates){
        shortlist_mask[id/8]|=(1u<<(id%8));
        int16_t yy[ADDRESS_RUNS];double sc=profile(addr,addresses[id],address_emission(addresses[id]),yy,nullptr);
        if(sc>best){second=best;best=sc;bestid=id;ba=a;bz=z;copy(yy,yy+ADDRESS_RUNS,bestY);}
        else second=max(second,sc);
      }
    }
  }
  *shortlist_size=totalCandidates;*margin=best-second;
  if(bestid<0)return 4;
  if(best<minimum_address_score||best-second<address_margin)return 5;
  if(bestid>=ALLOCATED)return 6;
  int begin=starts[ba]+rr[ba].n,end=starts[bz];string pay=body.substr(begin,end-begin);
  if(pay.size()>511)return 7;
  memcpy(payload,pay.data(),pay.size());payload[pay.size()]=0;*payload_size=pay.size();
  *identity=bestid;*orientation=orient;copy(bestY,bestY+ADDRESS_RUNS,address_y);return 0;
}
extern "C" double parse_payload(const char*raw,const double*w,const int*known_lengths,int16_t*ys,int*events,double*scores){
  auto obs=rle(string(raw));string pat="ACTACTACTACTACTACTACTACT";
  vector<array<double,21>> e(24);
  for(int i=0;i<24;i++)for(int y=0;y<=20;y++){
    int b=i%3;double p=0;
    if(known_lengths&&known_lengths[i]>=3)p=w[(b*13+known_lengths[i]-3)*21+y];
    else for(int s=0;s<4;s++)p+=.25*w[(b*13+states[s][b]-3)*21+y];
    e[i][y]=log(max(p,1e-12));
  }
  double score=profile(obs,pat,e,ys,events);
  for(int cell=0;cell<8;cell++)for(int s=0;s<4;s++){
    double ll=0;for(int b=0;b<3;b++){int y=ys[cell*3+b];if(y<0||y>20){ll-=30;continue;}ll+=log(max(w[(b*13+states[s][b]-3)*21+y],1e-12));}
    scores[cell*4+s]=ll;
  }
  return score;
}
'''
PRIOR_B85 = (
    'c-oy<X*5>v+eH#8DKaD!B{C09@|<nRtU-nfNupFTB+6K(kfF@;ERuPic|68UO$wzH6&cDDrR075e*aJJyWX|l_1+)Oea>0u!(RK^'
    '=eqX3|76d+IBI?ec7?C5)cbUSk)y2nFLygOmZXW>tm#D3pVZvNrdE{nZx^l}s09bpoXTsr9PF#|;;bwPN1w{0DP3ttOsnT{o9xmB'
    'eH0C9dv~Gmdbxuw-5f#gNd4NlzM6=(+2W_%?n2Bj_f{%o%MrCV!nu6LMoHZChAiUs5u{2!-wEBH@+9%%#P3<HZxv|%+v1UYw~N8k'
    '7iFKN+zPU5AnMgJ`<I6d*|&}#<hj~ltHP7U8v6vU(;Xj5zqcT6&p^20<~l@(`xh~`<$*8HDkd%>8fEf3)I#66VaCYkz56j^@LZtx'
    '_52}=PkByjmI4^yd^sRV_t8t@>EhwOe&;x1>^IYiU4F)d)&{zbKR7suFV6(tl8rh^NmP{HJ6%XIHe0;OhXqLvR-sdF4Q&eYrR`4j'
    '$@kD;c}H&e{v&uBB?K8N6=Rp=Hr1~A2Ar3E{##Y36@i;PhhLp&N90)ru5Q5=oC@Q1_|8*@b>FMnE9weBtFywg&MF>WhP8(FwO%;('
    'M@wx|-2#r9!R5tfN}yYRLvc2g6BNhY`t~|si9OQa>NdKS5g%Pj1-HC+BdB;Ps`Chf@<Vviw*43BF}2KdJBvu>tfvo^I-8TIS1Qo^'
    'sLKzlukwk&{(BmFq^oK0yEm{kb+H(<mTF5WkqyvMxod8$(gFjv{S--|Hf)H_-1^0=8KwOK$9n1NuyBZV)T+1;3<0z`+u4&*BW3y3'
    'z}yd+?@uKLU$@33&aF!}zqCQO`}ZY!1tEldzmVtr;1BUKSUMG1-Bg@)2hI$|5R&go{WIQ}5lugY_h~x_5W5*X4bE@=NKzO&9vp2G'
    'hGZ^xvnyY{(Bdc5d6PRFoArAPnHC@7w*DOvbd`crPLCMgPligMew6t0W;7M$oGPVl#kL<cB9nWX5bxsOoa0yn%bxIvaJfQcO^uDK'
    '-A;y&LUL+=YyjRnbsnmWwS!pspXw!=lkgOEPtMsQ1&jDe3y$P<c>8NW{z%v$@%h>8+Z4N8!ket6eM|c$VfOi1h};`RqBB*sd~P`c'
    'cia?s^{4D%GR@LqSmh1#t=(Ha3uB-}JLa4CCL2*K4ju}b<q&Mrd65`Ph9+gFM38Ya#7uP41V)-6koc*D?pq_e@=AX{BUgd(knL~t'
    '*?g4hMPByUo(#U-mu{%<y@y|cxw&_(uH&rEBXP=O1DL#8%9_&O2az;^UGE-jhVpbT2aU`aA?nxqd;Wb5k*|J!=w@dyVPtW5l~F7i'
    'A1$0KIFH`UPfLb*H}LyCyQfG@AabPWC5{uxcq*~G%VteJIDaZ7T02!DTy}1tbDRvhRvy)JZ<@eb+;ym|xe0W8_vvPqkl}nuHHm$;'
    '0$e>ReXKk4a7MQ(E1oq8=>gLreR}t>p5LrP@4f>Rcy8}L{OB|o-i|)GVJeR+T&a$Xlbi^!@3~H~nk5#5oh0WwI*D;1X~x{`aqvG-'
    'JT`swBwh*Ja@?SO5usu4p8S@&jaxo!aqabCII(GcCa-Zi=>Kwxq>2_{gj`E}6Rn0Nho?t}KqJ1fd)_;)-UOHKFW#DI6wteMibl-V'
    '<K=<9uKU=_!Fv5}Jef5I#K?|^({~bZ>*Lgyp(g=ok?**4Wc>~3DZe#1*ldJNC8MSmr3B^!_8oIq;sZ^J;6^|06~g)JD$oSb@w)!H'
    'w4eg9-r=`9zn?}h^Tp^R%-2yVJ-UG<?=EcqYUU+UVxS)6z3;kP7G8xoF&{T7!P&F6LldPnICHHf;kA4t#_IDLTM{W~y>!Lpv~wf+'
    '_BUvrzg>&y$1cn>lBIBDjh>0=e*}}EizcNH;^1MSr*SXWANdY5Gd5!mV9&6>Fg|JwX_kDF&axtOZkJyd;MfBynbgu#>)PNmt+TQp'
    'i=i&h>Nt*K``|^hn7fv^uW7Q;WX=_~)g0_GefPl^bEt;$G8qv@N{%L9bCBY4;TDZc83HHtv!YgN;l)4dZzkS|`g(VTnKzB78hv?h'
    'M1zbykrNaV)@n$-{AgINQw*(?LZ6kuO!y}DWaOWV!O+KZh9Z2_{t)+0^NoT7`ptZKl{rno&>KIfnW6!m0|82ApHpzaF`MHltr(*0'
    'G#L0AmC*U;b<}XG5rU#k&nD&Dql9j)|8Tnx$|4Mv$SINF7}3g8noNfx=}{yX3UO+hyKCk~1s3wwxNH8dL$u2BMY9Dmo(#CScPx`p'
    'oZ)^qN1z^Iv4b<_k18N~b5V@dz7S6HYT;ssGvK0<c{hV83P<M@77i-<A|*%l{G#A>Fld*z|Kzbi1lQ`|N<+es+Fy&c0$?0_`1pI-'
    'e)Le@&wmoog+IfrRTbMMa0_&H$Je-GP(Iaspe`7@rOX~TI3?ikt{<uAk3T|K#Qd%^e#OWMm5MEutHQ%G8waan>k+JSRxc@>j5QTm'
    '5lrjI$kQxnFM3pi5|Mt!3p!<xWVkX}qMC=krTTLTbE!bjv0$mpa5P7kS~y61W64GGC7Xi_wQqS`slAa7S2-D#K4&gyZ?*k(&_M!c'
    'N<VJ1_@#!H{RT_-Mb6=CPv9lnmFw6t!o1b?g%7;B1Qe^hBfxvYP)|xU4ZLcXT7FFB!u9pM3Q9}ir~UJ0OL{d5wyfF6|EwP9F0LG*'
    'Cu6Oo=`8zf9o`at3M7##&@ldR6QnIcMh4B}<eqFajD6Q@xRVTrQv%7o8lg~mIL?tN5sW|o_>c$ayBoi6x?mWGYc$g4th*5Jkd=7U'
    'ssR#Td-|2mWP(X+-LdC4eNZSb|0HhPX`KG0m{}$-fT82wXLAca5<HT;L0*3%3F>_IFmqQTOzu;Dh(7s9q7(JwIKzFDH1aUMd56>*'
    '63zc}1zC^&&YlUwfp1o<6y;aQDTr<6y#5r;a<>B?p6x_rmfC|`bYvJ#?KQC#&&EF1DD#q-d!QU9`?3t0L*m@lrm|^iXfVZ9H4iNi'
    '#65>=2mDHi;>2VDn`0M=uqn5X3+px$tA6K!X%T5Ut5th3Rh>kIKhP8XO5y*C^IyLdiRn{|(+<T=2xb4#@-njxug$X8pZVO4?60G@'
    ')r~qZzP9MyYZ)@`$%r<FHRs^YirW$Es0TPn`;0F#;VSlet=)EjMiCpToxe}@FrjdEe(~aL7ZDtNZ0EQ8cUSp!TChGyOz2j=-)l5X'
    'qPigb*x~&z&yRRWtFRaK3jSBztG>jXH3os-e`UJ#=OaA5)Q^o^jlkS_OJ3SGNRMjsxVd&A)gyAN)`chVWU6&^)FPvX=+2tq%EyZ0'
    '{j*<aBQVFwxBsbt1D+Re*hM+1g{Qph2Y0U9hOlLoacB2oLcWw0Ztv0v%frdx`l-f5;dQHSTE>lpvf1Yomxjvz^>0b1NUMK!aq#T_'
    '!27SSi^Fyg)!_8p80ZV}ECj3N;GqoH+A`GPTHeEn+Wl=vvQUt`tk#K}uUn$}E8D=iB1|tASdUNj0=D)%d9bUBJXd!&0;(~IVQg0&'
    'aZeD{$$lrm#k#K7_?-agL~qeqzWh!cGgl+Ezbc{njPdrz%GU|~k%trhFN6tRY8PKK+E1$1+&~v-<3-}9+U81*T_n@h7orUcPpBxj'
    'NhX5!%8nU-J#UbNKQ*g9jzzm+#lT6+LVT&T;6K^gfJncIa1Wz4oMY=?6=Lpy;w`Vox(uygm^eQ6Nx2p~4X1CJdgkDZr$V~q(Fl|i'
    '&NW(9PN=;TnX_L*50>f0{k+}65Y#TVZx#JXFp5Q0Z|iTQ`ewb(M|0j(9sk+sF``IN>+cbj{)4oNOMfYY^nq$VzUdtQkN-?Y`Q$=?'
    't)C6{oMoJ7Qul*DjbX9_TPm)uOSVhCT!O2n?6QxyHDbtOk7R;L8|D>h&rxdH5lmKb+oRKrMz+^VcVAS&+>z^<tW!1|beB7pZiGXb'
    'eO6$E`6h;i=!<XeGQjrAi+`VOlt68l$RuYV4Ne3zR-cgTBHW>Rw=f}$>WgYjmgw{d!MDF&CvRjSo(sHn_WMR5QS-{JuR^I12!ExP'
    '({~yifluoaXdU4ecFAi_Bow>>2PP$iGr`X=p4=T%j*ar^S5)^=pdwY!ep#{&r_BeRzSpDX6V^qk;y?;+Zf@P3uTufu1%oHqJF;+j'
    'u^@;>Fbu(AX=MLfPRMEK+w8t<fIOk#%^iQGpqy)Bf1;iq4g9Zdwk^LTq#0T0BCL`K>c7o@K*^k-_RUYM0s_RpJ~<6T$`)S-cd3HQ'
    't=yDH;{qa-3!Ag~JmD2;HadMR2Fe>=c`vWYh4hY=(-)|`JrCxe<KEeXH@wWb8{1paV7lw}lb#kBc@JtQw~-;k&j09SV=21&GM{Uw'
    'Wnd>SU0;yf1E?nJ(=v!Tf}O*DZ2f=%Mpc+<w0LE(-N<96d?zEQdX?ySF+fbYNk%kEXRhjY$FiLrRnHw%r<@M5wcZz1dRfRAh|Mr*'
    'odidPzQr3&dvsM*u78?y4~+ty&!?ZHU?u1B-ANQdTb;#f<F{Iz(EsiEv8xFxxA^^3`CFmd5?$;p){G?%R|#gudT4Bv<Mbzr@rmsT'
    'tv6>HIu)b$MQH@%$<l=d?o$p}AtsU%`t_;#Mp5&!fefOq9*ZzO&Wuv4b(Z|y?+M?DlKc?m!vFaB4D8d?m*<uifcEhFl~;VqC^*_U'
    'rT^_b{6`0M*-hM0G!{B7HWCie?>f&`v@+2)YV|%UuM|Df8{(Dg8*rm$Ha$qB8QGMaK^B=7j1P_eHnyXn=GIGTi=b*G@kgDENu}n!'
    'y|3i_C6l55?yu6kT>u8086B0y?4co&^2S_356r%|hCQmJaa!bDuq5|-Fk0Kha%GGVZ2u6NiK~YPURy_O!Ng`Oaq_SXjI#ZArfKP6'
    '|2(b!lBXRGmtVz;$9`Deabkz#wFHcA47T%(&&4kTjvo}hO6+3!!?u;V5p3-b%sz=WW87nh#0I)%P~KXbeoi4Hrks))EnES#*Kw|8'
    '$Uz9%wa4db98QfMi=@f$1+(~wjcLg>q<qUQUD=`w!*$u4KG{pbc_;f{_S#MOH+N*A*m7Nok~TfKJaQ+KI(K1P?}yFgTve!D3*C|_'
    'cmWM`tT&m`-QYgr-`pt_28-)r=cI3^qa^kCNN{}-oD<A6#Y}6EePw_wHQWfjftvzz-A(9E$zJx4q+sYv{o+7y9m-piM-so5;JNgs'
    'm-lKjp`p52T4N;=Nkc`O4l#P6pf-26@rKJtU6bLI=5iDwFOJUII>^A4JzxE!iw|r%<*oN=Q|C_gaOU=)cCf5rIo9XgiMGCZ?~{YA'
    'ICLserbew6a;Md<`v1+Q@_4y)q%jnSZ6}5HGh7FC*k6<p*M@6Eb8)^kA2iMf>(Og`CE{YY7A1=n5F}>K(k<TBM0Z!$&xZzrRNd(w'
    'w6lFlvL`Y33C_8Zs9U<-6hCPd4SRmW|CmQ{v8yWp5eyC=q9>~HLZ$1(fD#!;f=Q#7eB0o=C##Tw+SkPy0z%qZS`po^&B7N`3-hV4'
    '?fNFUD7Y$pdE6%wSvRJ?gf88}o)>rLJjD!gDf_8)1-As`L<h7P6;_BiZhwJ4&Fxg&1A^UaLI^oo+4ECv+SKRR#<;XK#3~jisWOr*'
    'bvZbkC;fw~YVE&!Y=J`6>bXv32)skml@6WEg69eS`N{3YnC{IaD+!XR`7@ufKcp4$i9Dytwe5H=_||8uMhgnV1f7SfYGHFtaiaZ7'
    'KGtX2yWduc#n_KwZvn?Uc=;wQB<GnKe4CVoh#7hKzKwq|F~JI1*?t3w><>g=dg(Xr@@yi*u5Nu7hb>Xb=j+^jU7Vn@DKXannWRCz'
    'xndqm5?M8f@#>^W-PIGQP%z7J+4X%3Lo5TI9ozd<Ocq+sqz@K@dG<RC|6~J%=qrj+vsz#-^O);ea2viq2>U7#(2UIBoad%xwHPS$'
    '`@q6ch^SPJy&+!`kP$XZ%lFg=iB7_8hv=<fKRnm!zF!T@J4@MhoOYm3#XKl9XpZ1tJF=xKhC=ZBb|#D)22pY9+l*`JuIl3P+-KI+'
    '{imE)C&}#}eMTVawA?4(<vUn(rD-qT9){005B4ZJK1A0gCV~CUrLbiq^9)}hV}6sn&!AijtkTm8pPIGeO4-HUI;|$G^sV)B5v#%V'
    '7s5ZYj}+jVLWjPrMItWS=$<_gdKVN9wj7gV*62HaLEC&q2mJdMPkSp1!&Uz*^Xo7g(9nd=U2%Uwt-r>IR$t00ug)UtFB4Q=YyG99'
    'h*kNLn4>ZI?qVGItSy!tK3-*0^#cF*@O4KsqcLd8=<(D#8&cdO;#DQ(_)y)mZQE-y_}*n{eyeLn=Odo*;6p7au-g(L$4r6cQ=@*r'
    'TUFTmTXt>v`#fw?E>!u>nFtB>RJAYueu(K_Hzk^E3;L_Mg6bK%U=B__=yzHQeNDWEX+likw><W(n)L&bukOmcz*I!VZ6=iMdWZN|'
    'r?EI>K^YtrHicvDr*{v^uA{1TS@YKC0OU+DsVnYFLU-Sd&5^}<X!GB!ONy_=TY+~oza<*M-=Q{QAm5CvAd4r>o0~x@bD*_qCnK<^'
    '<hUL+S9*IK>^;tti}STpLlwIcptxW&|E<9f+~+(Orbp}`bZE=e2%!&INx7_T4sz%&JV{@1X*0TmdPCY5##Z?)ERb&`_O9wwH~~{9'
    '@{M*n9>YqF^ykkN7jPqOYA!v`4GoHO9c(?J)ZASgLwrcXKIdGu#jQoyuHj0&w5moKqinS4wMLY47`?cX-UR=V-SnEDDG=toxJ7b9'
    '1B{jhGacK?@H(>Lay`dmByMEweV!kO7vk*xQ#w>%w!`_Zz&U$#Wd4m(=QV_^#@ez+j0bVD@P@Ij4hOb|RG)Sf_(A-0j$4_8h5J53'
    '$%hruOYw3yh%~^L9@3Xn%64#NJE^ND><go3FPXgBqS1XvX2o-B7Mk?bT-F<x;GtxKoe@(lO6bKO{S9bDBF_~2`FRR{;;3YDQ6rvw'
    'e&P}5UkjGFg_=u4CHSb8&^|Gig`vXrt#kb`xNXoY8GZCF9G-GZIBMBrflS-fyXO>0ZfaU$?efqt*i<<Ec?UA=Ry%A8=7I_OwBs@;'
    'zWB(;Fi!(6=GKxtOc%huW$M}U6lbXITOM@Z9s&-PEOELINl4<*Zyry{!IYtbY+iU7G}ss8!Y|k1<kz3E?>!o^O{OzMZ!ZM`ue@8v'
    '9m)99Q5(R<R1KxeZ|)rYSp*dkyFc>}AEM*h9)}7_6e6rI96CAdgNnO<S=LwCLF7=y*4^h$;m$XS^hH4x{M(mOVV9C57W!c~>hd|h'
    'HinYGbm6K_p8)l{d-pY7kJv(^_P5C=V{epgNeyKxjlksc*$+*3((yO_-h<x~g=k*vEQz7&QL?<)=fcf;{AT(an5o`~6dwm(%_EIS'
    '`(z#xXIYQk*33>Zp%u8C#?<qZs}Rpl9ZxGTO~(=2jE8UhBCtW~=K{B=H~v_bd+qhJ#U0L9y2q`|Ah$XIWYZB>$hjDp%L5wS=r@5e'
    'GN^7i#5AUP6w@~?G_u_+U~<zuhsV(w<)#lNw77#Xy!=PoF(@9P+iL4xEN3C9GwhG}{bIysBtMPxtU|g_W}S#2HLt|yom==w#wHCJ'
    'R#6o)%)FE&ZF_5QqkyQS+fxQQzTQJ0@8?1&EGdC)XDW^oIy1izh7a=*A&#n^P_Hfvd~fgcpZT6aofD%Ujl8zN<ap>_vHQZHVeLKf'
    '@RSOCFHk=EA2G(o8{fN&W$n?u?L<PSwKwea6-I|r!{PgD?36Mk7105$wwD`oaWP>m^~mc|h>sgYA6!?1$Fvg0jOQB=6VN%9c$AC|'
    'zChQbdiAhLr%5&@s<5-{*N3Ja>O8qBasKSqY+U@YXMEF4B51y7p4qW{AHKHEMl<&V@L#QY)${'
)
PRIOR_SHA256 = '18ee23e3298579db78b1a02df6e26aa9cf43809a6adf1b7bfa38aa79fa973f27'


def load_prior():
    """Load the exact public probability array, stored as little-endian float64."""
    raw = zlib.decompress(base64.b85decode(PRIOR_B85))
    if hashlib.sha256(raw).hexdigest() != PRIOR_SHA256:
        raise ValueError("Embedded public prior is damaged")
    prior = np.frombuffer(raw, dtype="<f8").reshape(3, 13, 21).copy()
    if not np.isfinite(prior).all() or (prior < 0).any():
        raise ValueError("Invalid embedded public prior")
    if not np.allclose(prior.sum(-1), 1, atol=1e-12):
        raise ValueError("Embedded public prior rows are not normalized")
    return prior


def address_pattern(message):
    bits = [0] * 17
    positions = [i for i in range(1, 16) if i not in (1, 2, 4, 8)]
    for shift, position in enumerate(positions):
        bits[position] = (message >> shift) & 1
    for parity in (1, 2, 4, 8):
        bits[parity] = sum(bits[i] for i in range(1, 16) if i & parity) % 2
    bits[16] = sum(bits[1:16]) % 2
    word = sum(bits[i] << (i - 1) for i in range(1, 17)) ^ 0xF6CF
    state = 0
    pattern = ""
    for shift in range(16):
        state = (state + 1 + ((word >> shift) & 1)) % 3
        pattern += BASES[state]
    return pattern


class Parser:
    def __init__(self, prior, directory):
        compiler = next((shutil.which(name) for name in ("c++", "clang++", "g++")
                         if shutil.which(name)), None)
        if compiler is None:
            raise RuntimeError("A C++17 compiler is required on PATH")
        source = directory / "parser.cpp"
        library = directory / ("parser.dylib" if sys.platform == "darwin" else "parser.so")
        source.write_text(PARSER_SOURCE)
        command = [compiler, "-O3", "-std=c++17", "-fPIC",
                   "-dynamiclib" if sys.platform == "darwin" else "-shared",
                   str(source), "-o", str(library)]
        compiled = subprocess.run(command, capture_output=True, text=True, check=False)
        if compiled.returncode:
            raise RuntimeError("Native parser compilation failed\n" + compiled.stderr)
        self.library = ct.CDLL(str(library))
        self.dp = ct.POINTER(ct.c_double)
        self.ip = ct.POINTER(ct.c_int)
        self.sp = ct.POINTER(ct.c_int16)
        self.library.init_parser.argtypes = [ct.c_char_p, self.dp, self.ip]
        self.library.init_parser.restype = None
        self.library.parse_structure.argtypes = [
            ct.c_char_p, self.ip, self.ip, self.ip, self.dp, ct.c_char_p,
            self.ip, self.sp, ct.POINTER(ct.c_uint8), ct.c_int, ct.c_double, ct.c_double,
        ]
        self.library.parse_structure.restype = ct.c_int
        self.library.parse_payload.argtypes = [ct.c_char_p, self.dp, self.ip,
                                               self.sp, self.ip, self.dp]
        self.library.parse_payload.restype = ct.c_double
        self.prior = np.ascontiguousarray(prior, dtype=np.float64)
        patterns = "".join(address_pattern(i) for i in range(ADDRESS_CAPACITY))
        self.library.init_parser(patterns.encode(), self.prior.ctypes.data_as(self.dp),
                                 STATES.ctypes.data_as(self.ip))

    def structure(self, sequence):
        identity, orientation, count, length = (ct.c_int() for _ in range(4))
        margin = ct.c_double()
        body = ct.create_string_buffer(512)
        observations = np.zeros(ADDRESS_RUNS, np.int16)
        mask = np.zeros(MASK_BYTES, np.uint8)
        code = self.library.parse_structure(
            sequence.encode(), ct.byref(identity), ct.byref(orientation), ct.byref(count),
            ct.byref(margin), body, ct.byref(length), observations.ctypes.data_as(self.sp),
            mask.ctypes.data_as(ct.POINTER(ct.c_uint8)), CONFIG["max_primer_edits"],
            CONFIG["address_margin"], CONFIG["minimum_address_score"],
        )
        return code, identity.value, orientation.value, margin.value, body.value, observations

    def payload(self, body, probabilities, known=None):
        probabilities = np.ascontiguousarray(probabilities, dtype=np.float64)
        observations = np.full(24, -1, np.int16)
        events = np.zeros(4, np.int32)
        scores = np.zeros((8, 4), np.float64)
        lengths = None if known is None else np.ascontiguousarray(known, dtype=np.int32)
        score = self.library.parse_payload(
            body, probabilities.ctypes.data_as(self.dp),
            None if lengths is None else lengths.ctypes.data_as(self.ip),
            observations.ctypes.data_as(self.sp), events.ctypes.data_as(self.ip),
            scores.ctypes.data_as(self.dp),
        )
        return score, observations.reshape(8, 3), events, scores


def fastq_records(path):
    with path.open("rb") as raw:
        compressed = raw.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="ascii", newline=None) as handle:
        ordinal = 0
        while True:
            name = handle.readline()
            if not name:
                break
            ordinal += 1
            sequence = handle.readline()
            plus = handle.readline()
            quality = handle.readline()
            if not sequence or not plus or not quality:
                raise ValueError(f"Truncated FASTQ record {ordinal}")
            name = name.rstrip("\r\n")
            sequence = sequence.rstrip("\r\n")
            plus = plus.rstrip("\r\n")
            quality = quality.rstrip("\r\n")
            if not name.startswith("@") or len(name) < 2 or not plus.startswith("+"):
                raise ValueError(f"Invalid FASTQ delimiters in record {ordinal}")
            if not sequence or len(sequence) != len(quality):
                raise ValueError(f"Invalid FASTQ sequence or quality length in record {ordinal}")
            if any(ord(character) < 33 or ord(character) > 126 for character in quality):
                raise ValueError(f"Invalid FASTQ quality character in record {ordinal}")
            yield sequence.upper()


def adapt(parser, records, prior):
    counts = np.zeros_like(prior)
    fit_ids, qc_ids, qc = set(), set(), []
    for identity, body, address_observations, margin in records:
        if identity < DATA_OLIGOS or margin < 3:
            continue
        row = np.asarray([(identity - DATA_OLIGOS + j) % 4 for j in range(8)])
        lengths = STATES[row]
        score, observed, _, _ = parser.payload(body, prior, lengths)
        if score < -100 or (observed < 0).any():
            continue
        if identity < DATA_OLIGOS + 24:
            fit_ids.add(identity)
            for cell in range(8):
                for base in range(3):
                    counts[base, int(lengths[cell, base]) - 3, int(observed[cell, base])] += 1
            pattern = address_pattern(identity)
            # skip run zero, primer/address boundary may coalesce
            for j in range(1, ADDRESS_RUNS):
                if 0 <= address_observations[j] <= 20:
                    counts[BASES.index(pattern[j]), 2, int(address_observations[j])] += 1
        else:
            qc_ids.add(identity)
            qc.append((lengths, observed))
    estimated = prior.copy()
    shrinkage = CONFIG["prior_shrinkage"]
    for base in range(3):
        for length in (3, 5, 7, 10):
            count = counts[base, length - 3]
            if count.sum() >= CONFIG["minimum_pilot_row_count"]:
                estimated[base, length - 3] = (
                    (count + .5 + shrinkage * prior[base, length - 3])
                    / (count.sum() + 10.5 + shrinkage)
                )
    nll_prior, nll_estimated, observations = 0., 0., 0
    for lengths, observed in qc:
        for cell in range(8):
            for base in range(3):
                length, value = int(lengths[cell, base]) - 3, int(observed[cell, base])
                observations += 1
                nll_prior -= np.log(max(prior[base, length, value], 1e-12))
                nll_estimated -= np.log(max(estimated[base, length, value], 1e-12))
    delta = (nll_estimated - nll_prior) / observations if observations else None
    return estimated, {
        "fit_identities": len(fit_ids),
        "qc_identities": len(qc_ids),
        "qc_run_observations": observations,
        "qc_prior_nll": nll_prior / observations if observations else None,
        "qc_adapted_nll": nll_estimated / observations if observations else None,
        "qc_nll_change": delta,
        "calibration_failure": bool(len(fit_ids) < 4 or len(qc_ids) < 2 or (delta is not None and delta > .5)),
        "warning": bool(len(fit_ids) < 12 or len(qc_ids) < 4 or (delta is not None and delta > .2)),
        "prior_shrinkage": shrinkage,
        "per_bin_pseudocount": .5,
        "minimum_row_count": CONFIG["minimum_pilot_row_count"],
    }




def fastq_parents(path):
    path = Path(path)
    with path.open('rb') as raw:
        compressed = raw.read(2) == b'\x1f\x8b'
    opener = gzip.open if compressed else open
    with opener(path, 'rt', encoding='ascii') as handle:
        for ordinal, line in enumerate(handle):
            if ordinal % 4:
                continue
            fields = line.rstrip('\r\n').split()
            tags = {}
            for field in fields[1:]:
                if field.count(':') >= 2:
                    name, _, value = field.split(':', 2)
                    tags[name] = value
                elif '=' in field:
                    name, value = field.split('=', 1)
                    tags[name] = value
            yield tags.get('pi') or fields[0][1:]

def extract_fastq(path, output):
    prior = load_prior()
    rows, rejected = [], Counter()
    with tempfile.TemporaryDirectory(prefix='lovelace-parser-') as directory:
        parser = Parser(prior, Path(directory))
        for index, (sequence, parent) in enumerate(zip(fastq_records(Path(path)), fastq_parents(path))):
            if set(sequence) - set('ACGT'):
                rejected['ambiguous_nucleotides'] += 1
                continue
            code, identity, orientation, margin, body, observed_address = parser.structure(sequence)
            if code:
                rejected[str(code)] += 1
                continue
            score, counts, events, scores = parser.payload(body, prior)
            if score < CONFIG['minimum_payload_score'] or ((counts < 1) | (counts > 20)).any():
                rejected['payload'] += 1
                continue
            rows.append((identity, orientation, counts, parent))
    np.savez_compressed(output, identity=np.array([row[0] for row in rows], dtype=np.int64),
                        orientation=np.array([row[1] for row in rows], dtype=np.int64),
                        counts=np.array([row[2] for row in rows], dtype=np.int64).reshape(-1, 8, 3),
                        raw_parent_id=np.array([row[3] for row in rows]))
    return dict(accepted=len(rows), rejected=dict(rejected),
                extraction='original standalone ordinary run profile parser',
                current_paper_observations=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fastq', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(extract_fastq(args.fastq, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
