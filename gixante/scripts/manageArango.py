import sys, pickle, time

from gixante.utils.arango import log, getCollection, fromNewbies, splitPartition, count, checkPivotCount, cfg
import gixante.utils.parsing as parsing

# runtime args
if len(sys.argv) < 2: sys.argv.append("news")
if len(sys.argv) < 3: sys.argv.append("250")
if len(sys.argv) < 4: sys.argv.append("5000")

collectionName = sys.argv[1]
partitionSize = int(sys.argv[2])
batchSize = int(sys.argv[3])

partitionQ = "FOR doc in {0}Newbies LIMIT {1} RETURN"
pivCountErrTol = .01

# check indices and other stuff
#collection = getCollection(collectionName)

# load data
weights, voc, coordModel = pickle.load(open(cfg['dataDir'] + '/forManager.pkl', 'rb'))

# configure parser for validation
parsing.configForCollection(collectionName)
parser = parsing.Parser()

###

# CAUTION: this is NOT threadsafe!

while True:
    # assign new docs to partitions
    if count(collectionName + 'Newbies') > batchSize:
        newCounts = fromNewbies("LIMIT {0}".format(batchSize), collectionName, voc, weights, coordModel, parser)
        
        # now iterate over all the fat partitions
        log.info("Splitting large partitions...")
        [ splitPartition(pid, voc, weights, collectionName, parser, partitionSize) for pid, count in newCounts.items() if count > partitionSize*2 ]
    
    else:
        log.info("Not enough new documents - will catch up with some housekeeping...")

        samplePivCountErr = 1
        while samplePivCountErr > pivCountErrTol:
            res = checkPivotCount(collectionName)
            samplePivCountErr = res[1]*res[2]

        time.sleep(60)
