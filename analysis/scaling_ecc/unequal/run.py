"""compare predeclared digit-significance protection at equal physical budgets."""

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
from ..outer.core import (K, HEADER, make_library, encode_stripe,
    decode_stripe, decode_library, parse_envelope)
from ..sequence.codecs import Codec
from ..sequence.payloads import load_frozen_payloads, unpack_levels, reconstruct_lovelace
from tsv_records import write_output

ALLOCATION = {'metadata':8, 'highest digit':16, 'middle digit':8, 'lowest digit':0}
PROFILE = dict(version='digitplanes-v1', digits='highest to lowest', allocation=ALLOCATION,
    uniform_parity=8, packet_bytes=2, data_packets=32)


def plane_bytes(payload):
    labels = unpack_levels(payload,6,(512,))
    return [np.packbits(((labels[:,None] >> [shift + 1, shift]) & 1).reshape(-1)).tobytes()
            for shift in (4,2,0)]


def unpack_plane(values):
    digits=[]
    for value in values[:128]:
        digits.extend([None] * 4 if value is None else [(value >> shift) & 3 for shift in (6,4,2,0)])
    return digits


def join_planes(planes):
    digits=[unpack_plane(plane) for plane in planes]
    labels=[None if any(plane[i] is None for plane in digits) else
            16 * digits[0][i] + 4 * digits[1][i] + digits[2][i] for i in range(512)]
    bits=[]
    for label in labels:
        bits.extend([None] * 6 if label is None else [(label >> shift) & 1 for shift in (5,4,3,2,1,0)])
    values=[]
    for i in range(0,len(bits),8):
        byte=bits[i:i+8]
        values.append(None if None in byte else sum(v << shift for v,shift in zip(byte,range(7,-1,-1))))
    return labels, values


def libraries(obj):
    metadata=dict(obj['metadata'], protection_profile=PROFILE)
    uniform=make_library(obj['payload'],'lovelace',2,8,metadata)
    offset=len(uniform.envelope)-len(obj['payload'])
    segments=[uniform.envelope[:offset]]+plane_bytes(obj['payload'])
    packets, groups, identity=[],[],0
    for (role,r),segment in zip(ALLOCATION.items(),segments):
        data_count=((len(segment)+63)//64)*32
        data=segment.ljust(data_count*2,b'\0')
        for stripe in range(data_count//32):
            words=encode_stripe([data[i:i+2] for i in range(stripe*64,(stripe+1)*64,2)],r)
            indices=[]
            for slot,word in enumerate(words):
                packets.append(dict(identity=identity,role=role,stripe=stripe,slot=slot,is_parity=slot>=32,
                    parity_packets=r,bytes=word))
                indices.append(identity)
                identity+=1
            groups.append(dict(role=role,r=r,indices=indices))
    expected=sum(len(g['indices']) for g in groups)
    assert expected==len(uniform.packets)
    return uniform,dict(packets=packets,groups=groups,offset=offset,metadata=metadata,
        segments=segments,segment_lengths=list(map(len,segments)))


def decode_unequal(estimates,layout):
    recovered={role:[] for role in ALLOCATION}
    failed=0
    for group in layout['groups']:
        values,bad=decode_stripe([estimates[i] for i in group['indices']],group['r'],2)
        recovered[group['role']].extend(value for packet in values for value in packet)
        failed+=bad
    labels,body=join_planes([recovered[role] for role in list(ALLOCATION)[1:]])
    header=recovered['metadata'][:layout['offset']]
    parsed=parse_envelope(header+body,'lovelace',8,2)
    parsed.update(labels=labels,scored_payload=body,failed_columns=failed,scored_metadata=header)
    return parsed


def damage(packets,condition,probability,rng):
    output=[]
    for packet in packets:
        values=list(packet['bytes'])
        if rng.random()<probability:
            if condition=='packet dropout':
                values=[None,None]
            else:
                values=[value ^ int(rng.integers(1,256)) for value in values]
        output.append(values)
    return output


def metrics(obj,metadata,decoded,payload_values):
    truth=unpack_levels(obj['payload'],6,(512,))
    bits=[]
    for value in payload_values:
        bits.extend([None]*8 if value is None else [(value>>shift)&1 for shift in range(7,-1,-1)])
    labels=[]
    for i in range(0,len(bits),6):
        part=bits[i:i+6]
        labels.append(None if None in part else sum(v<<shift for v,shift in zip(part,range(5,-1,-1))))
    wrong=sum(v is not None and v!=t for v,t in zip(labels,truth))
    missing=sum(v is None for v in labels)
    exact=missing==0 and wrong==0
    metarecovered=decoded['metadata']==metadata
    rmse=None
    if not any(v is None for v in payload_values) and isinstance(decoded['metadata'],dict):
        try:
            image,_,_=reconstruct_lovelace(bytes(payload_values),decoded['metadata'])
            expected,_,_=reconstruct_lovelace(obj['payload'],obj['metadata'])
            rmse=float(np.sqrt(np.mean((image-expected)**2)))
        except (ValueError,TypeError,KeyError,OverflowError):
            pass
    resolved=[(v,t) for v,t in zip(labels,truth) if v is not None]
    return dict(exact_quantized_stream=exact,exact_stored_object=exact and metarecovered,
        residual_component_errors=wrong,missing_components=missing,
        residual_component_error_fraction=(wrong+missing)/512,
        resolved_component_rmse=float(np.sqrt(np.mean([(2*(v-int(t))/63)**2 for v,t in resolved]))) if resolved else None,
        image_rmse=rmse,metadata_recovered=metarecovered,integrity_accepted=decoded['accepted'],
        incorrect_accepted=decoded['accepted'] and not (exact and metarecovered),
        declared_failure=not decoded['accepted'],failed_columns=decoded['failed_columns'])


def write_table(path,rows):
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    HERE.mkdir(parents=True,exist_ok=True)
    objects=[obj for obj in load_frozen_payloads() if obj['codec']=='lovelace']
    spec=dict(saved_before_run=True,source_objects=[dict(name=obj['name'],payload_sha256=hashlib.sha256(obj['payload']).hexdigest(),sources=obj['sources']) for obj in objects],
        profile=PROFILE,allocation_basis='numerical digit weights sixteen, four and one and required reconstruction metadata, before examining outcomes',
        byte_symbol_order='plane bytes contain four two-bit digits in descending bit order',
        metadata='complete protected outer header and JSON include native reconstruction header and digests',
        packet_boundary='two bytes and thirty-two data packets per stripe, each significance plane is independently padded',
        matched_budget='same data packet and parity packet counts, address count, per-strand length and padding as uniform r8',
        initial_control='error-free sequence readout and exact outer recovery required before corruption',
        experiment='outer-code packet test',conditions=['packet dropout','unknown erroneous packet'],
        probabilities=[0,.05,.10,.20,.40],realizations=20,seed=12102026,
        errors='both bytes in selected data or parity packet xor an independently uniform nonzero byte, no oracle erasure',
        sequencing='same sixteen requested reads per designed strand, no molecular simulation claimed',
        missing_metrics='missing components remain missing, image RMSE available only when the payload and decoded reconstruction metadata permit reconstruction')
    write_output(HERE/'analysis_specification.tsv', spec)
    codec=Codec('lovelace')
    rows,costs,controls,mappings=[],[],[],[]
    for obj in objects:
        uniform,unequal=libraries(obj)
        pristine=decode_library([list(p['bytes']) for p in uniform.packets],'lovelace',8,2)
        pristine_uep=decode_unequal([list(p['bytes']) for p in unequal['packets']],unequal)
        assert pristine['payload']==pristine_uep['payload']==obj['payload']
        assert pristine['accepted'] and pristine_uep['accepted']
        for arm,packets in [('uniform',uniform.packets),('unequal',unequal['packets'])]:
            sequences=[codec.encode(p['identity'],p['bytes']) for p in packets]
            decoded=codec.decode_reads(sequences,list(range(len(packets))))
            assert not decoded.erased.any()
            np.testing.assert_array_equal(decoded.packets,np.array([list(p['bytes']) for p in packets],np.uint8))
            path=HERE/f"{obj['name']}_{arm}_data_and_parity.fasta"
            path.write_text(''.join(f">{obj['name']}_{arm}_{p['identity']:04d}_{'parity' if p['is_parity'] else 'data'}\n{seq}\n" for p,seq in zip(packets,sequences)))
            controls.append(dict(image=obj['name'],arm=arm,error_free_sequence_exact=True,
                error_free_outer_exact=True,sequence_count=len(sequences)))
            data=sum(not p['is_parity'] for p in packets)
            parity=len(packets)-data
            costs.append(dict(image=obj['name'],arm=arm,stored_payload_bytes=len(obj['payload']),
                protected_metadata_bytes=unequal['offset'],data_packets=data,parity_packets=parity,
                data_padding_bytes=data*2-len(uniform.envelope),designed_nucleotides=sum(map(len,sequences)),
                data_nucleotides=data*298,added_parity_nucleotides=parity*298,
                address_nucleotides=len(sequences)*80,total_requested_reads=16*len(sequences),
                requested_sequenced_nucleotides=16*sum(map(len,sequences)),budget_mismatch_nt=0))
            for p,seq in zip(packets,sequences):
                mappings.append(dict(image=obj['name'],arm=arm,identity=p['identity'],
                    role=p.get('role','uniform serialized stream'),stripe=p['stripe'],slot=p['slot'],
                    parity_packets=p.get('parity_packets',8),is_parity=p['is_parity'],
                    sequence_length_nt=len(seq),packet_hex=p['bytes'].hex()))
            for condition in spec['conditions']:
                for probability in spec['probabilities']:
                    for draw in range(20):
                        realization_seed=spec['seed']+draw+1000*spec['probabilities'].index(probability)+100000*spec['conditions'].index(condition)
                        rng=np.random.default_rng(realization_seed)
                        estimates=damage(packets,condition,probability,rng)
                        if arm=='uniform':
                            result=decode_library(estimates,'lovelace',8,2)
                            payload_values=result['values'][unequal['offset']:unequal['offset']+384]
                        else:
                            result=decode_unequal(estimates,unequal)
                            payload_values=result['scored_payload']
                        rows.append(dict(image=obj['name'],arm=arm,condition=condition,probability=probability,
                            realization=draw,seed=realization_seed,experiment='outer-code packet test',
                            added_nucleotides=parity*298,total_nucleotides=sum(map(len,sequences)),
                            requested_sequenced_nucleotides=16*sum(map(len,sequences)),
                            **metrics(obj,unequal['metadata'],result,payload_values)))
    write_table(HERE/'realizations.csv',rows)
    write_table(HERE/'physical_budget.csv',costs)
    write_table(HERE/'packet_mapping.csv',mappings)
    write_output(HERE/'controls.tsv', controls)
    summary=[]
    for image in [obj['name'] for obj in objects]:
        for arm in ('uniform','unequal'):
            for condition in spec['conditions']:
                for probability in spec['probabilities']:
                    selected=[r for r in rows if (r['image'],r['arm'],r['condition'],r['probability'])==(image,arm,condition,probability)]
                    available=[r['image_rmse'] for r in selected if r['image_rmse'] is not None]
                    summary.append(dict(image=image,arm=arm,condition=condition,probability=probability,
                        realizations=len(selected),exact_quantized_stream=sum(r['exact_quantized_stream'] for r in selected),
                        exact_stored_object=sum(r['exact_stored_object'] for r in selected),metadata_recovered=sum(r['metadata_recovered'] for r in selected),
                        component_error_fraction_mean=float(np.mean([r['residual_component_error_fraction'] for r in selected])),
                        image_rmse_mean=float(np.mean(available)) if available else None,image_rmse_available=len(available)))
    write_table(HERE/'summary.csv',summary)
    print(json.dumps(dict(realizations=len(rows),matched_budget=True,error_free_controls=len(controls))))


if __name__=='__main__':
    main()
