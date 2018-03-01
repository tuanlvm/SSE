import numpy as np
import sys
from sam.corpus.corpus import CorpusReader
from sse.sse import SSE

import sam.log as log
CORPUS_FILENAME = sys.argv[2];
num_topics = int(sys.argv[1]);
reader = CorpusReader(CORPUS_FILENAME, data_series='sam')
model = SSE(reader,T=num_topics)

count = 0
while count<100:
    print 'loop ',count
    model.run_one_iteration()
    count = count + 1
    
model.write_topics(num_top_words=10, num_bottom_words=10);
model.write_topic_weights_arff();
