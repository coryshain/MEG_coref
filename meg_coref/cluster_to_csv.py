import sys

clust = None
print('word,subtlex_freq,dist,clusters,cs,bl,ef')
for line in sys.stdin:
    if line.startswith('Cluster'):
        clust = line.strip().split(' ')[1][:-1]
    elif not line.startswith('Num clusters'):
        line = line.strip().split()
        if len(line) == 3:
            w, freq, dist = line
            print(','.join([w, freq, dist, clust, '', '', '']))