"""recompute native lovelace accounting and finite-length extensions."""
import csv
import math
from pathlib import Path
import numpy as np
import pandas as pd
from stream_format import strand, address_word as native_address_word
from ..sequence.codecs import Codec, address_bits
from tsv_records import read_output, write_output

HERE = Path(__file__).resolve().parent
CONFIG = read_output(HERE.parent / "configuration.tsv")
ADDRESS_LENGTHS = tuple(CONFIG["address_lengths"])
LIMITS = tuple(CONFIG["strand_limits"])
PARITIES = tuple(CONFIG["parity_packets"])
METADATA_BYTES = CONFIG["metadata_bytes"]
SEED = CONFIG["scaling_seed"]

def write_csv(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (HERE / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def write_record(name, obj):
    write_output(HERE / name, obj)

def address_word(message, length):
    parity_positions = {1 << i for i in range((length - 1).bit_length())}
    data_positions = [i for i in range(1, length) if i not in parity_positions]
    if not 0 <= message < 1 << len(data_positions):
        raise ValueError('message exceeds shortened SECDED capacity')
    bits = [0] * (length + 1)
    for shift, position in enumerate(data_positions):
        bits[position] = (message >> shift) & 1
    for parity in parity_positions:
        bits[parity] = sum(bits[i] for i in range(1, length) if i & parity) % 2
    bits[length] = sum(bits[1:length]) % 2
    mask = 0xF6CF & ((1 << length) - 1)
    return sum(bits[i] << (i - 1) for i in range(1, length + 1)) ^ mask

def transition(bits, start='A'):
    state = 'ACT'.index(start)
    result = []
    for bit in bits:
        state = (state + 1 + int(bit)) % 3
        result.append('ACT'[state])
    return ''.join(result)

def address_dna(identity, length):
    word = address_word(identity, length)
    return ''.join(base * 5 for base in transition((word >> i) & 1 for i in range(length)))

def minimum_distance(words):
    return min(bin(left ^ right).count("1") for i, left in enumerate(words) for right in words[i + 1:])

def native_records():
    records = []
    for line in (HERE / "inputs/Lovelace_S12_480.fasta").read_text().splitlines():
        if line.startswith(">"):
            records.append([line[1:].split()[0], ""])
        else:
            records[-1][1] += line.strip()
    with np.load(HERE / "inputs/design_reference.npz", allow_pickle=False) as source:
        digits, runs = source["expected_cell_states"], source["expected_cell_runs"]
    states = np.asarray([(3, 7, 10), (3, 10, 7), (7, 3, 10), (7, 10, 3)])
    assert np.array_equal(states[digits], runs)
    validation = []
    for identity, (name, sequence) in enumerate(records):
        assert strand(identity, digits[identity], states) == sequence
        assert len(sequence) == 298
        validation.append(dict(codec="lovelace", strand=name, nt=len(sequence), reconstructed=True))
    write_csv("native_sequence_validation.csv", validation)
    return records


def baseline(records):
    total = sum(len(sequence) for _, sequence in records)
    row = dict(codec="lovelace", format="native S12 quantized images", stored_object="six-bit quantized real and imaginary Fourier components for images A and B", logical_payload_bits=2 * 16 * 16 * 2 * 6, unique_stored_objects=2, designed_strands=len(records), designed_nt=total, min_strand_nt=min(len(seq) for _, seq in records), median_strand_nt=float(np.median([len(seq) for _, seq in records])), max_strand_nt=max(len(seq) for _, seq in records), payload_region_nt=480 * 160, useful_payload_region_nt=384 * 160, primer_nt=50 * len(records), address_nt=80 * len(records), marker_nt=8 * len(records), header_nt=32 * 160, digest_nt=32 * 160, pilots_nt=32 * 160, padding_nt=0, parity_nt=0, payload_bits_per_payload_nt=6144/(384*160), payload_bits_per_synthesized_nt=6144/total, nominal_field_bits_per_strand=16, current_recovery_endpoint="quantized component stream exact equality with all six historical digest outcomes retained separately", current_recovery_population="A and B under three existing readout arms", decoder_public_knowledge="S12 header field format address SECDED cell state mapping quantizer and Fourier crop", decoder_candidate_knowledge="no expected image or coefficients supplied to decoder", existing_redundancy="address SECDED and 32 known pilots; current source conjugate-format gate rejects encoder-wide validity so partner inference is not a guaranteed redundant coefficient channel", length_rule="8 complete 20-nt cells per strand with six-bit components crossing 16-bit strand boundaries", source="frozen S12 480 FASTA design reference and FORMAT.md")
    assert sum(row[key] for key in ("useful_payload_region_nt", "primer_nt", "address_nt", "marker_nt", "header_nt", "digest_nt", "pilots_nt", "padding_nt", "parity_nt")) == total
    for key in ("primer", "address", "marker", "header", "digest", "pilots", "padding", "parity"):
        row[key + "_fraction"] = row[key + "_nt"] / total
    recovery = pd.read_csv(HERE / "inputs/lovelace_native_six_checksum_outcomes.tsv", sep="\t")
    row.update(current_recovery_numerator=int(recovery.stream_exact.sum()), current_recovery_denominator=len(recovery), current_recovery_population="A and B under control position and position_weighted decoders", current_recovery_unit="quantized stream outputs across existing arms", current_checksum_passes=int(recovery.checksum_pass.sum()))
    write_csv("native_format_accounting.csv", [row])
    write_csv("native_recovery_endpoints.csv", [dict(codec="lovelace", population=r["image"], arm=r["arm"], endpoint="exact quantized stream recovery", correct=int(r["stream_exact"]), total=1, checksum_pass=r["checksum_pass"], header_exact=r["header_exact"], payload_exact=r["payload_exact"]) for r in recovery.to_dict("records")])
    return row


def construct_addresses():
    rows, constructions = [], []
    for length in ADDRESS_LENGTHS:
        parity_count = (length - 1).bit_length()
        dimension = length - parity_count - 1
        generators = [address_word(1 << i, length) ^ address_word(0, length) for i in range(dimension)]
        syndrome_columns = list(range(1, length))
        assert len(set(syndrome_columns)) == length - 1
        assert all(bin(g).count("1") % 2 == 0 for g in generators)
        exact_minimum = None
        if length <= 24:
            exact_minimum = min(bin(address_word(value, length) ^ address_word(0, length)).count("1") for value in range(1, 1 << dimension))
            assert exact_minimum >= 4
        for value in range(min(128, 1 << dimension)):
            dna = address_dna(value, length)
            assert len(dna) == 5 * length
            assert all(a != b for a, b in zip("A" + dna[::5], dna[::5]))
        volume = sum(math.comb(length, i) for i in range(4))
        lower = ((1 << length) + volume - 1) // volume
        upper = min((1 << length) // (length + 1), 1 << (length - 3))
        rows.append(dict(codec="common proposed extension", address_length_bits=length, address_nt=5*length, distance_metric="binary transition-word Hamming distance", minimum_distance=exact_minimum or 4, evidence_class="constructed shortened extended Hamming code generator", verified_size=(1 << dimension) if exact_minimum is not None else "", constructed_size=1 << dimension, theoretical_lower_bound=lower, theoretical_upper_bound=upper, dimension_bits=dimension, construction="systematic LSB-first shortened Hamming plus even overall parity and XOR mask", DNA_edit_distance_guarantee=False, codeword_enumeration="complete" if exact_minimum is not None else "generator proven with first 128 sequences checked"))
        constructions.append(dict(length_bits=length, dimension_bits=dimension, generators=[f"{g:0{length}b}" for g in generators], xor_mask=f"{address_word(0,length):0{length}b}", capacity=1 << dimension, verified_enumerated_minimum=exact_minimum, guaranteed_minimum_hamming=4, proof="parity-check columns are distinct and nonzero so no weight-two word exists and overall parity excludes weights one and three"))
    native = next(row for row in rows if row["address_length_bits"] == 16).copy()
    native.update(codec="lovelace", evidence_class="verified native complete public codebook", verified_size=2048, constructed_size="", construction="same encoder as native common.address_word checked for all 2048 words", deployed_strands=480)
    assert all(address_word(i, 16) == native_address_word(i) for i in range(2048))
    rows.append(native)
    write_csv("address_capacity.csv", rows)
    write_record("address_constructions.tsv", constructions)
    return rows


def layout(length, address_bits):
    primers, markers = 50, 8
    available = length - primers - markers - 5 * address_bits
    positions = max(0, available // 20)
    width = positions // 4
    positions = width * 4
    payload_nt = positions * 20
    actual = primers + markers + address_bits * 5 + payload_nt
    return dict(codec="lovelace", strand_limit_nt=length, address_length_bits=address_bits, address_nt=address_bits*5, primer_nt=primers, marker_nt=markers, available_payload_budget_nt=max(0,available), usable_payload_positions=positions, position_unit="two-bit 20-nt cell", packet_bytes=width, payload_bits_per_strand=width*8, worst_case_payload_nt=payload_nt, fixed_padding_nt=0, worst_case_strand_nt=actual, unused_limit_nt=max(0,length-actual), exceeds_limit_nt=max(0,actual-length), fits_all_allowed_packet_payloads=width>0 and actual<=length, fit_class="guaranteed byte-aligned fit", format="proposed representation allocation", native_decoder_support=False, payload_bits_per_payload_nt=width*8/payload_nt if payload_nt else 0, payload_bits_per_complete_nt=width*8/actual if width else 0, address_fraction=5*address_bits/actual, padding_fraction=0, header_fraction=0, parity_fraction=0, component_boundary_rule="six-bit components concatenate across strands without forced alignment")


def capacity_tables(records):
    rows = [layout(length, address) for length in LIMITS for address in ADDRESS_LENGTHS]
    lengths = np.asarray([len(sequence) for _, sequence in records])
    for length in LIMITS:
        rows.append(dict(codec="lovelace", strand_limit_nt=length, format="native unchanged library reference", fit_class="exact released sequence validation", address_length_bits=16, payload_bits_per_strand=16, total_native_strands=len(lengths), native_strands_within_limit=int(np.sum(lengths<=length)), native_strands_exceeding_limit=int(np.sum(lengths>length)), native_min_nt=int(np.min(lengths)), native_max_nt=int(np.max(lengths)), native_decoder_support=True, fits_all_allowed_packet_payloads=bool(np.all(lengths<=length))))
    write_csv("length_limited_capacity.csv", rows)
    return rows


def validate_proposed(capacities):
    rng = np.random.default_rng(SEED)
    # consume the earlier draws so the rng state matches
    for width in CONFIG["preceding_random_byte_widths"]:
        rng.bytes(width)
    records, fasta = [], []
    for row in capacities:
        if row.get("fit_class") != "guaranteed byte-aligned fit" or not row["packet_bytes"]:
            continue
        width, address_length = row["packet_bytes"], row["address_length_bits"]
        codec = Codec("lovelace", address_length, width)
        for label, payload in [("zero", bytes(width)), ("maximal", bytes([255])*width), ("independent binary", rng.bytes(width))]:
            for identity in (0, min(127, codec.address_capacity - 1)):
                seq = codec.encode(identity, payload)
                assert len(seq) <= row["strand_limit_nt"] and len(seq) <= row["worst_case_strand_nt"]
                assert sum(bit << i for i, bit in enumerate(address_bits(identity, address_length))) == address_word(identity, address_length)
                identifier = f'lovelace_generic_L{row["strand_limit_nt"]}_a{address_length}_{label.replace(" ", "_")}_{identity}'
                fasta.append(f">{identifier}\n{seq}\n")
                records.append(dict(strand=identifier, codec="lovelace", format=row["format"], address_length_bits=address_length, strand_limit_nt=row["strand_limit_nt"], packet_bytes=width, payload_class=label, payload_hex=payload.hex(), sequence_nt=len(seq), analytical_maximum_nt=row["worst_case_strand_nt"], length_verified=True, address_word_verified=True))
    (HERE / "proposed_length_validation.fasta").write_text("".join(fasta))
    write_csv("proposed_sequence_validation.csv", records)
    return records

def file_tables(capacities, addresses):
    book={row['address_length_bits']:int(row['constructed_size']) for row in addresses if row['codec']=='common proposed extension'}
    rows=[]
    for row in capacities:
        if row.get('fit_class')!='guaranteed byte-aligned fit' or not row['packet_bytes']:
            continue
        width=row['packet_bytes']
        capacity=book[row['address_length_bits']]
        for parity in PARITIES:
            stripe_packets=32+parity
            usable_stripes=capacity//stripe_packets
            supported_bytes=max(0,usable_stripes*32*width-METADATA_BYTES)
            theory = next(r for r in addresses if r['codec']=='common proposed extension' and r['address_length_bits']==row['address_length_bits'])
            lower_file = max(0,theory['theoretical_lower_bound']//stripe_packets*32*width-METADATA_BYTES)
            upper_file = max(0,theory['theoretical_upper_bound']//stripe_packets*32*width-METADATA_BYTES)
            native_capacity = 2048 if row['codec']=='lovelace' and row['address_length_bits']==16 else 0
            native_file = max(0,native_capacity//stripe_packets*32*width-METADATA_BYTES)
            for name,size in [('1 KiB',1<<10),('1 MiB',1<<20),('1 GiB',1<<30)]:
                useful_packets=math.ceil((size+METADATA_BYTES)/width)
                stripes=math.ceil(useful_packets/32)
                data_packets=stripes*32
                parity_packets=stripes*parity
                total=data_packets+parity_packets
                padding=data_packets*width-size-METADATA_BYTES
                total_nt=total*row['worst_case_strand_nt']
                rows.append(dict(codec=row['codec'],format=row['format'],scenario=name,payload_bytes=size,protected_file_metadata_bytes=METADATA_BYTES,strand_limit_nt=row['strand_limit_nt'],address_length_bits=row['address_length_bits'],constructed_address_capacity=capacity,packet_bytes=width,parity_packets_per_stripe=parity,data_packets_per_stripe=32,required_data_packets=data_packets,required_parity_packets=parity_packets,required_strands=total,stripes=stripes,padding_bytes=padding,total_designed_nt=total_nt,total_designed_nt_kind='worst-case guarantee for all byte values',payload_bits_per_complete_nt=size*8/total_nt,supported_file_size_after_metadata_ecc_bytes=supported_bytes,supported_by_constructed_codebook=total<=capacity,supported_by_theoretical_upper_bound=total<=next(r['theoretical_upper_bound'] for r in addresses if r['codec']=='common proposed extension' and r['address_length_bits']==row['address_length_bits']),metadata_fraction=METADATA_BYTES/(data_packets*width),parity_fraction=parity_packets/total,padding_fraction=padding/(data_packets*width),address_fraction=row['address_fraction'],requested_reads_per_strand=16,total_requested_reads=16*total,total_requested_sequence_nt_worst_case=16*total_nt,arbitrary_message_decoder_demonstrated=False,accounting_scenario=True,metadata_schema='56-byte core envelope plus compact empty JSON object; all metadata included in RS input',bound_note='construction size is a lower bound on maximum codebook capacity'))
                rows[-1].update(verified_native_address_capacity=native_capacity,verified_native_book_file_capacity_bytes=native_file,theoretical_lower_file_capacity_bytes=lower_file,theoretical_upper_file_capacity_bytes=upper_file,verified_native_book_applied_to_proposed_file_layout=True,native_fixed_object_codec_can_decode_this_file=False)
    write_csv('file_size_scaling.csv',rows)
    return rows

def required_length_reference():
    rows = []
    for codec,width in [('lovelace',2)]:
        for label,size in [('1 KiB',1<<10),('1 MiB',1<<20),('1 GiB',1<<30)]:
            for parity in PARITIES:
                data = math.ceil((size+METADATA_BYTES)/(32*width))*32
                total = data//32*(32+parity)
                address = next(n for n in range(8,65) if 1 << (n-(n-1).bit_length()-1) >= total)
                required_nt = 218 + 5*address
                rows.append(dict(codec=codec,scenario=label,payload_bytes=size,packet_bytes=width,parity_packets_per_stripe=parity,protected_file_metadata_bytes=METADATA_BYTES,required_strands=total,minimum_transmitted_address_bits=address,constructed_address_capacity=1 << (address-(address-1).bit_length()-1),required_complete_strand_nt=required_nt,total_designed_nt_worst_case=total*required_nt,exceeds_200_nt=required_nt>200,exceeds_250_nt=required_nt>250,exceeds_300_nt=required_nt>300,format='proposed fixed packet widths from executed adapters',decoder_implemented_at_this_address_length=False,accounting_scenario=True))
    write_csv('minimum_length_file_scenarios.csv',rows)
    return rows

def main():
    records = native_records()
    baseline(records)
    addresses = construct_addresses()
    capacities = capacity_tables(records)
    proposed = validate_proposed(capacities)
    files = file_tables(capacities, addresses)
    required_length_reference()


if __name__ == "__main__":
    main()
