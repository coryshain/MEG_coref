filter = set()
with open('output/stim/filter.txt', 'r') as f:
    for line in f:
        filter.add(line.strip().split()[0])

with open('output/stim/all_preds_filtered.txt', 'w') as o:
    with open('output/stim/all_preds.txt', 'r') as f:
        for line in f:
            w = line.strip().split()
            if w:
                w = w[0]
                if w in filter:
                    o.write(line)
