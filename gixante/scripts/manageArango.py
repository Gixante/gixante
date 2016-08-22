import sys, pickle, time

sys.path.append('/home/bean/Code/Python')
from gixante.utils.arango import log, getCollection, assignBatchToPartitions, splitPartition, count

# runtime args
if len(sys.argv) < 2: sys.argv.append("news")
if len(sys.argv) < 3: sys.argv.append("250")
if len(sys.argv) < 4: sys.argv.append("5000")

collectionName = sys.argv[1]
partitionSize = int(sys.argv[2])
batchSize = int(sys.argv[3])

# check indices and other stuff
#collection = getCollection(collectionName)

# load data
weights, voc, coordModel = pickle.load(open('/home/bean/catapi_data/forManager.pkl', 'rb'))

###

# CAUTION: this is NOT threadsafe!

partitionQ = "FOR doc in {0}Newbies LIMIT {1} RETURN"
nDocsQ = "FOR doc in news FILTER doc.partition == {0} LIMIT %d COLLECT WITH COUNT INTO c RETURN c" % max(batchSize, partitionSize*2)

while True:
    # assign new docs to partitions
    if count(collectionName + 'Newbies') > batchSize:
        newCounts = assignBatchToPartitions(partitionQ.format(collectionName, batchSize), voc, weights, collectionName, coordModel)
        
        # now iterate over all the fat partitions
        log.info("Splitting large partitions...")
        [ splitPartition(pid, voc, weights, collectionName, partitionSize) for pid, count in newCounts.items() if count > partitionSize*2 ]
    
    else:
        log.info("Not enough new documents - waiting a minute...")
        time.sleep(60)
