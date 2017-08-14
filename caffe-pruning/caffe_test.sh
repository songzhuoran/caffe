#!/usr/bin/env sh

./build/tools/caffe test -model examples/cifar10/cifar10_quick_train_test.prototxt -weights examples/cifar10/test_caffemodel_pruning02/bit6sigma_test6_2000_0.caffemodel -gpu 0 2>&1 | tee a.txt

if [ $? -eq 0 ] ; then 

echo "iter: "> Test_accuracy.txt
grep -c 'accuracy =' a.txt >> Test_accuracy.txt
echo "interval: ">> Test_accuracy.txt
grep 'test_interval:' a.txt | cut -d ' ' -f 2 >> Test_accuracy.txt
echo ' ' >> Test_accuracy.txt
grep 'accuracy =' a.txt | cut -d '#' -f 2 | cut -d ' ' -f 2,3,4 >> Test_accuracy.txt


echo "iter:" >Train_loss.txt
grep -c 'Train net output #0: loss = ' a.txt >> Train_loss.txt
echo "interval:" >> Train_loss.txt
grep 'display' a.txt  | cut -d ' ' -f 2 >> Train_loss.txt
echo ' ' >> Train_loss.txt
grep 'Train net output #0: loss = ' a.txt | cut -d ':' -f 5 | cut -d '(' -f 1 | cut -d ' ' -f 2,3,4 >> Train_loss.txt

echo "iter:" >Test_loss.txt
grep -c 'Test net output #1: loss = ' a.txt >> Test_loss.txt
echo "interval:">> Test_loss.txt
grep 'test_interval:' a.txt | cut -d ' ' -f 2 >> Test_loss.txt
echo ' ' >> Test_loss.txt
grep 'Test net output #1: loss = ' a.txt | cut -d '#' -f 2 | cut -d ' ' -f 2,3,4 >> Test_loss.txt


# mv Test_accuracy.txt ~/lilai/myDL/clothes/mymodel/model/
# mv Train_loss.txt ~/lilai/myDL/clothes/mymodel/model
# mv Test_loss.txt ~/lilai/myDL/clothes/mymodel/model

fi