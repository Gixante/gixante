
import sys
from tqdm import tqdm
from numpy import sort, where

from gixante.utils.http import getCreatedTs, log
from gixante.utils.arango import database, getPivotIds

if len(sys.argv) < 2: sys.argv.append("fwd")

direction = sys.argv[1]

missingTsQ = "FOR doc IN news FILTER doc.partition == {0} FILTER doc.createdTs == NULL OR (doc.createdTs < 0 AND doc.createdTs > -37) RETURN doc"
collection = database.col('news')
pids = sort(getPivotIds('news'))

if direction != 'fwd': pids = pids[::-1] 

for pid in pids:
    docs = list(database.execute_query(missingTsQ.format(pid)))
    nDocs = len(docs)
    log.info("Partition {0:,} ({2} of {3}) has {1} missing timestamps".format(pid, nDocs, where(pids==pid)[0][0], len(pids)))
    res = [ collection.update_document(doc['_key'], {'createdTs': getCreatedTs(doc['URL'])}) for doc in tqdm(docs) ]
