// same extraction rules as the public python code
#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using Run = std::pair<char,int>;

static bool act(char b) { return b=='A'||b=='C'||b=='T'; }
static int val(char b) { return b=='A'?0:b=='C'?1:2; }

static std::string rc(const std::string& s) {
  std::string r(s.rbegin(),s.rend());
  for(char& c:r) c=c=='A'?'T':c=='T'?'A':c=='C'?'G':c=='G'?'C':c;
  return r;
}

static std::vector<Run> runs(const std::string& s) {
  std::vector<Run> r;
  for(char b:s) {
    if(!act(b)) continue;
    if(!r.empty()&&r.back().first==b) ++r.back().second;
    else r.emplace_back(b,1);
  }
  bool changed=true;
  while(changed&&r.size()>2) {
    changed=false;
    std::vector<Run> merged={r[0]};
    size_t i=1;
    while(i<r.size()) {
      if(i+1<r.size()&&r[i].second==1&&merged.back().first==r[i+1].first) {
        merged.back().second+=1+r[i+1].second;
        i+=2; changed=true;
      } else merged.push_back(r[i++]);
    }
    r.swap(merged);
  }
  return r;
}

static int tail(const std::string& b) {
  int n=b.size(),i=n-1;
  while(i>=0&&b[i]=='G') --i;
  if(n-i-1>=3) return i+1;
  int start=std::max(0,n-30);
  for(int j=start;j+2<n;++j) if(b[j]=='G'&&b[j+1]=='G'&&b[j+2]=='G') return j;
  return -1;
}

static bool trim(const std::string& read,const std::string& f,const std::string& r,std::string& body) {
  const std::string fc=f.substr(5,15),rr=r.substr(5,15);
  for(int orientation=0;orientation<2;++orientation) {
    const std::string s=orientation?rc(read):read;
    auto start=s.substr(0,60).find(fc);
    if(start==std::string::npos) continue;
    int p=std::max(0,int(start)-5+int(f.size()));
    int t=std::max(0,int(s.size())-60);
    auto stop=s.substr(t).find(rr);
    int end=stop==std::string::npos?s.size():t+int(stop)-5;
    if(end>p) { body=s.substr(p,end-p); return true; }
  }
  return false;
}

static bool bits(const std::vector<Run>& r,int count,std::vector<int>& result) {
  int previous=0;
  for(int i=0;i<count;++i) {
    int v=val(r[i].first),step=(v-previous+3)%3;
    if(step==0) return false;
    result.push_back(step-1);previous=v;
  }
  return true;
}

static int address(const std::vector<int>& observed) {
  int n=observed.size(),syndrome=0,overall=0;
  std::vector<int> b(n+1,0);
  for(int i=1;i<=n;++i) { b[i]=observed[i-1]^((0xF6CF>>(i-1))&1);overall^=b[i]; }
  for(int i=1;i<n;++i) if(b[i]) syndrome^=i;
  if(syndrome&&!overall) return -1;
  if(syndrome) { if(syndrome>=n) return -1;b[syndrome]^=1; }
  else if(overall) b[n]^=1;
  int identity=0,shift=0;
  for(int i=1;i<n;++i) if(i&(i-1)) identity|=b[i]<<(shift++);
  return identity;
}

static void decode(const std::string& read,int mode,int n,int width,int32_t& identity,uint8_t* observations,uint8_t& status) {
  const std::string f="CCCAATTAATACGTTGCCTACTCCG";
  const std::string r="CGCTATCAGAACAAGAGAAGCACTA";
  identity=-1;status=0;
  std::string body;
  if(!trim(read,f,r,body)) return;
  status=1;
  if(body.size()<100||body.size()>400) return;
  std::vector<Run> a;
  int end=0;
  {
    status=2;
    const std::string region=body.substr(0,5*n+30);
    int marker=-1,stop=-1;
    for(int i=0;i+2<int(region.size());++i) if(region[i]=='G'&&region[i+1]=='G'&&region[i+2]=='G') {
      marker=i;stop=i+3;while(stop<int(region.size())&&region[stop]=='G') ++stop;break;
    }
    if(marker<0) return;
    a=runs(body.substr(0,marker));end=stop;
  }

  status=3;
  if(int(a.size())<n) return;
  std::vector<int> ab;
  if(!bits(a,n,ab)) return;
  identity=address(ab);
  if(identity<0) return;
  status=4;
  int t=tail(body);
  if(t<=end) return;
  auto payload=runs(body.substr(end,t-end));
  {
    if(int(payload.size())!=12*width) return;
    for(int i=0;i<12*width;++i) {
      if(payload[i].first!="ACT"[i%3]) return;
      observations[i]=std::min(payload[i].second,20);
    }
  }
  status=5;
}

extern "C" void decode_batch(const char* blob,const uint64_t* offsets,uint64_t count,int mode,int n,int width,int32_t* identities,uint8_t* observations,uint8_t* statuses) {
  for(uint64_t i=0;i<count;++i) decode(std::string(blob+offsets[i],offsets[i+1]-offsets[i]),mode,n,width,identities[i],observations+i*48,statuses[i]);
}
