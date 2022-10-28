set -x
cp configs/cbert_jigsaw_attention_based.config run.config
PROCESSED_DATA_DIR=processed_data_attention_based
python3 preprocess_attention_based.py raw_data/jigsaw/sentiment.train.0 label jigsaw.train.0 jigsaw $PROCESSED_DATA_DIR
python3 preprocess_attention_based.py raw_data/jigsaw/sentiment.dev.0 label jigsaw.dev.0 jigsaw $PROCESSED_DATA_DIR
python3 preprocess_attention_based.py raw_data/jigsaw/sentiment.test.0 label jigsaw.test.0 jigsaw $PROCESSED_DATA_DIR
python3 preprocess_attention_based.py raw_data/jigsaw/sentiment.train.1 label jigsaw.train.1 jigsaw $PROCESSED_DATA_DIR
python3 preprocess_attention_based.py raw_data/jigsaw/sentiment.dev.1 label jigsaw.dev.1 jigsaw $PROCESSED_DATA_DIR
python3 preprocess_attention_based.py raw_data/jigsaw/sentiment.test.1 label jigsaw.test.1 jigsaw $PROCESSED_DATA_DIR
rm $PROCESSED_DATA_DIR/jigsaw/train.data.label
rm $PROCESSED_DATA_DIR/jigsaw/dev.data.label
rm $PROCESSED_DATA_DIR/jigsaw/test.data.label
cat $PROCESSED_DATA_DIR/jigsaw/jigsaw.train.*.data.label >> $PROCESSED_DATA_DIR/jigsaw/train.data.label
cat $PROCESSED_DATA_DIR/jigsaw/jigsaw.dev.*.data.label >> $PROCESSED_DATA_DIR/jigsaw/dev.data.label
cat $PROCESSED_DATA_DIR/jigsaw/jigsaw.test.*.data.label >> $PROCESSED_DATA_DIR/jigsaw/test.data.label
/home2/tgv2002/miniconda3/envs/py37/bin/python3 shuffle.py $PROCESSED_DATA_DIR/jigsaw/train.data.label
/home2/tgv2002/miniconda3/envs/py37/bin/python3 shuffle.py $PROCESSED_DATA_DIR/jigsaw/dev.data.label
cp $PROCESSED_DATA_DIR/jigsaw/train.data.label.shuffle $PROCESSED_DATA_DIR/jigsaw/train.data.label
cp $PROCESSED_DATA_DIR/jigsaw/dev.data.label.shuffle $PROCESSED_DATA_DIR/jigsaw/dev.data.label
