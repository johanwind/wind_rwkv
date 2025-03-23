import subprocess, sys
for alg in ['smallhead', 'longhead', 'chunked', 'bighead_fp32', 'bighead_bf16', 'fla']:
    print(alg)
    for forward in [False,True]:
        for (batchsz,modeldim,headsz,seqlen) in [(8,4096,64,4096), (8,4096,128,4096), (8,4096,256,4096), (1,4096,256,4096*8)]:
            out = subprocess.run(f"python speed_test.py --alg {alg} --batchsz {batchsz} --modeldim {modeldim} --headsz {headsz} --seqlen {seqlen} "+"--forward"*forward, shell=True, capture_output=True, text=True).stdout
            try:
                out = out.split('\n')
                gb = float(out[1].split()[2])
                ms = float(out[2].split()[1])
            except Exception:
                ms = gb = float('nan')
                print(out, file=sys.stderr)
            print((batchsz,modeldim,headsz,seqlen,forward), ms, gb)
