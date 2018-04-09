
from LocalitySensitiveHashing import *
import csv
import sys

csvfilename1 = sys.argv[1] # the data
csvfilename2 = sys.argv[2] # the labels
csvfilename3 = sys.argv[3] # the combined file for LSH algorithm
#
try :
    f1 = open(csvfilename1,'r')
    f2 = open(csvfilename2,'r')
    f3 = open(csvfilename3,'w')
except:
    print("file open failed")
    exit(1)
#
X_t = list(csv.reader(f1))
Y_t = list(csv.reader(f2))
# print(X_t[1])
N = len(X_t)
D = len(X_t[0])
# print(N)
# XY_t = [0]*N
content = [0]*N
j = 0;
for i in range(0,N,2):
    content[j] = 'sample' + str(Y_t[i][0]) + '_' + str(j) +',' + ','.join(X_t[i])
    f3.write(content[j]+'\n')
    j = j + 1
# writer = csv.writer(open(csvfilename3, 'w'), delimiter=',')
# writer.writerows(content)
#
##  Clustering_with_LSH_with_sample_based_merging.py

##  The script demonstrates clustering using the neighbors produced by the basic 
##  LSH algorithm.  The neighborhoods are first coalesced on the basis of shared
##  data samples (which, in general, as I have mentioned elsewhere, is NOT a safe 
##  thing to do with the output of the LSH algorithm).  Subsequently, if the number
##  of clusters thus created (also referred to as 'similarity groups' in this
##  module) exceeds the expected number specified in the call to the module 
##  constructor, the module pools together the samples in the smallest of the 
##  clusters that are in excess.  Each sample in the pool is then assigned to the
##  retained clusters on the basis of closeness of the distance between the sample
##  and the cluster means.

##  Call syntax:
##
##         Clustering_with_LSH_with_sample_based_merging.py

#
lsh = LocalitySensitiveHashing(
           datafile = csvfilename3,
           dim = D,
           r = 2,                # number of rows in each band for r-wise AND in each band
           b = 5,               # number of bands for b-wise OR over all b bands
           expected_num_of_clusters = 10,
      )
lsh.get_data_from_csv()
lsh.show_data_for_lsh()
lsh.initialize_hash_store()
lsh.hash_all_data()
lsh.display_contents_of_all_hash_bins_pre_lsh()

similarity_groups = lsh.lsh_basic_for_neighborhood_clusters()
coalesced_similarity_groups = lsh.merge_similarity_groups_with_coalescence( similarity_groups )

merged_similarity_groups = lsh.merge_similarity_groups_with_l2norm_sample_based( coalesced_similarity_groups )

lsh.evaluate_quality_of_similarity_groups( merged_similarity_groups )

print( "\n\nWriting the clusters to file 'clusters.txt'" )
lsh.write_clusters_to_file( merged_similarity_groups, "clusters.txt" )