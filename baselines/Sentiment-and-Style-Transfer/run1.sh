main_operation=$1
main_function=$2
main_data=$3
main_dict_num=$4
main_dict_thre=$5
main_dev_num=$6

if [ "$main_data" = "amazon" ]; then
  batch_size=64
elif [ "$main_data" = "Jigsaw" ]; then
  batch_size=64
else
  batch_size=256
fi

main_function_orgin=$main_function
if [ "$main_function" = "DeleteOnly" ]; then
main_function=label
elif [ "$main_function" = "DeleteAndRetrieve" ]; then
main_function=orgin
elif [ "$main_function" = "RetrieveOnly" ]; then
main_function=orgin
fi

if [ "$main_data" = "yelp" ]; then
if [ ! -n "$main_dict_num" ]; then
main_dict_num=7000
main_dict_thre=15
fi
main_dev_num=4000
elif [ "$main_data" = "imagecaption" ]; then
if [ ! -n "$main_dict_num" ]; then
main_dict_num=3000
main_dict_thre=5
fi
main_dev_num=1000
elif [ "$main_data" = "amazon" ]; then
if [ ! -n "$main_dict_num" ]; then
main_dict_num=10000
main_dict_thre=5.5
fi
main_dev_num=2000
elif [ "$main_data" = "Jigsaw" ]; then
if [ ! -n "$main_dict_num" ]; then
main_dict_num=10000
main_dict_thre=5.5
fi
main_dev_num=2000
fi

main_category=sentiment
main_category_num=2
#configure
preprocess_tool_path=src/tool/
data_path=data/
train_file_prefix=${main_category}.train.
dev_file_prefix=${main_category}.dev.
test_file_prefix=${main_category}.test.
orgin_train_file_prefix=${data_path}${main_data}/$train_file_prefix
orgin_dev_file_prefix=${data_path}${main_data}/$dev_file_prefix
orgin_test_file_prefix=${data_path}${main_data}/$test_file_prefix
train_data_file=train.data.${main_function}
test_data_file=test.data.${main_function}
dict_data_file=zhi.dict.${main_function}


#preprocess test data
cp run-bash/* ./
line_num=$(wc -l < $train_data_file)
vt=$main_dev_num
eval $(awk 'BEGIN{printf "train_num=%.6f",'$line_num'-'$vt'}')
test_num=$main_dev_num
vaild_num=0
eval $(awk 'BEGIN{printf "train_rate=%.6f",'$train_num'/'$line_num'}')
eval $(awk 'BEGIN{printf "vaild_rate=%.6f",'$vaild_num'/'$line_num'}')
eval $(awk 'BEGIN{printf "test_rate=%.6f",'$test_num'/'$line_num'}')
for((i=0;i<$main_category_num;i++))
do
        python ${preprocess_tool_path}preprocess_test.py ${orgin_test_file_prefix}${i} ${train_file_prefix}${i} $main_function $main_dict_num $main_dict_thre ${test_file_prefix}${i}
        python ${preprocess_tool_path}filter_template_test.py ${test_file_prefix}${i} ${main_function}
        python ${preprocess_tool_path}filter_template.py ${train_file_prefix}${i} ${main_function}
done

for((i=0;i<$main_category_num;i++))
do
	THEANO_FLAGS="${THEANO_FLAGS}" python src/main.py ../model $train_data_file $dict_data_file src/aux_data/stopword.txt src/aux_data/embedding.txt $train_rate $vaild_rate $test_rate ChoEncoderDecoderDT generate_emb ${test_file_prefix}${i}.template.${main_function} $batch_size
  THEANO_FLAGS="${THEANO_FLAGS}" python src/main.py ../model $train_data_file $dict_data_file src/aux_data/stopword.txt src/aux_data/embedding.txt $train_rate $vaild_rate $test_rate ChoEncoderDecoderDT generate_emb ${train_file_prefix}${i}.template.${main_function} $batch_size
done

for((i=0;i<$main_category_num;i++))
do
	python ${preprocess_tool_path}find_nearst_neighbot_all.py $i $main_data ${main_function}
	python ${preprocess_tool_path}form_test_data.py ${test_file_prefix}${i}.template.${main_function}.emb.result
done


#test process
for((i=0;i<$main_category_num;i++))
do 
THEANO_FLAGS="${THEANO_FLAGS}" python src/main.py ../model $train_data_file $dict_data_file src/aux_data/stopword.txt src/aux_data/embedding.txt $train_rate $vaild_rate $test_rate ChoEncoderDecoderDT generate_b_v_t ${test_file_prefix}${i}.template.${main_function}.emb.result.filter $batch_size
done
if [ "$main_function_orgin" = "RetrieveOnly" ]; then
python ${preprocess_tool_path}get_retrieval_result.py
for((i=0;i<$main_category_num;i++))
do
cp ${test_file_prefix}${i}.retrieval ${test_file_prefix}${i}.${main_function_orgin}.$main_data
done
exit
fi
for((i=0;i<$main_category_num;i++))
do
	python ${preprocess_tool_path}build_lm_data.py ${orgin_train_file_prefix}${i} ${train_file_prefix}${i}
	python ${preprocess_tool_path}shuffle.py ${train_file_prefix}${i}.lm
	cp ${train_file_prefix}${i}.lm.shuffle ${train_file_prefix}${i}.lm
	python ${preprocess_tool_path}create_dict.py ${train_file_prefix}${i}.lm ${train_file_prefix}${i}.lm.dict
done
for((i=0;i<$main_category_num;i++))
do
	vaild_num=$i
	eval $(awk 'BEGIN{printf "vaild_rate=%.10f",'$vaild_num'/'$line_num'}')
	THEANO_FLAGS="${THEANO_FLAGS}" python src/main.py ../model ${train_file_prefix}${i}.lm ${train_file_prefix}${i}.lm.dict src/aux_data/stopword.txt src/aux_data/embedding.txt $train_rate $vaild_rate $test_rate ChoEncoderDecoderLm train $batch_size
done
vaild_num=0
i=0
eval $(awk 'BEGIN{printf "vaild_rate=%.10f",'$vaild_num'/'$line_num'}')
THEANO_FLAGS="${THEANO_FLAGS}" python src/main.py ../model ${train_file_prefix}${i}.lm ${train_file_prefix}${i}.lm.dict src/aux_data/stopword.txt src/aux_data/embedding.txt $train_rate $vaild_rate $test_rate ChoEncoderDecoderLm generate_b_v_t_v ${test_file_prefix}1.template.${main_function}.emb.result.filter.result $batch_size
vaild_num=1
i=1
eval $(awk 'BEGIN{printf "vaild_rate=%.10f",'$vaild_num'/'$line_num'}')
THEANO_FLAGS="${THEANO_FLAGS}" python src/main.py ../model ${train_file_prefix}${i}.lm ${train_file_prefix}${i}.lm.dict src/aux_data/stopword.txt src/aux_data/embedding.txt $train_rate $vaild_rate $test_rate ChoEncoderDecoderLm generate_b_v_t_v ${test_file_prefix}0.template.${main_function}.emb.result.filter.result $batch_size

for((i=0;i<$main_category_num;i++))
do
        python ${preprocess_tool_path}get_final_result.py ${test_file_prefix}${i}.template.${main_function}.emb.result.filter.result.result ${i}
	cp ${test_file_prefix}${i}.template.${main_function}.emb.result.filter.result.result.final_result ${test_file_prefix}${i}.${main_function_orgin}.$main_data

done

fi
