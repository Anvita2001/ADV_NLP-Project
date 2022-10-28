set -x
PROJECTPATH=/home2/tgv2002/MLM_transfer/
cp configs/bert_yelp_attention_based.config run.config
PYTHONPATH=$PROJECTPATH python3 fine_tune_bert.py
cp configs/cbert_yelp_attention_based.config run.config
PYTHONPATH=$PROJECTPATH python3 fine_tune_cbert.py
#PYTHONPATH=$PROJECTPATH $PYTHON_HOME/python test_tools/yang_test_tool/cls_wd.py
PYTHONPATH=$PROJECTPATH python3 fine_tune_cbert_w_cls.py
