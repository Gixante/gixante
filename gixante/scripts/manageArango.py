import sys, pickle, time

from gixante.utils.arango import log, getCollection, assignBatchToPartitions, splitPartition, count, checkPivotCount, cfg

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
weights, voc, coordModel = pickle.load(open(cfg['dataDir'] + '/forManager.pkl', 'rb'))

###

# CAUTION: this is NOT threadsafe!
partitionQ = "FOR doc in {0}Newbies LIMIT {1} RETURN"
pivCountErrTol = .05

while True:
    # assign new docs to partitions
    if count(collectionName + 'Newbies') > batchSize:
        newCounts = assignBatchToPartitions(partitionQ.format(collectionName, batchSize), voc, weights, collectionName, coordModel)

        
        # now iterate over all the fat partitions
        log.info("Splitting large partitions...")
        [ splitPartition(pid, voc, weights, collectionName, partitionSize) for pid, count in newCounts.items() if count > partitionSize*2 ]
    
    else:
        log.info("Not enough new documents - will catch up with some housekeeping...")
        
        samplePivCountErr = 1
        while samplePivCountErr > pivCountErrTol:
            res = checkPivotCount(collectionName)
            samplePivCountErr = res[1]*res[2]
        
        time.sleep(60)
