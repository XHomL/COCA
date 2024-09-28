python source_features.py --dataset officehome --img-augmentation none --source art --num-classes 15
python source_features.py --dataset officehome --img-augmentation none --source clipart --num-classes 15
python source_features.py --dataset officehome --img-augmentation none --source product --num-classes 15
python source_features.py --dataset officehome --img-augmentation none --source realworld --num-classes 15
#
python source_features.py --dataset domainnet --img-augmentation none --source painting --num-classes 200
python source_features.py --dataset domainnet --img-augmentation none --source real --num-classes 200
python source_features.py --dataset domainnet --img-augmentation none --source sketch --num-classes 200
#
python source_features.py --dataset visda --img-augmentation none --source train --num-classes 9
#
python source_features.py --dataset officehome --img-augmentation flip --img-views 2 --source art --num-classes 15
python source_features.py --dataset officehome --img-augmentation flip --img-views 2 --source clipart --num-classes 15
python source_features.py --dataset officehome --img-augmentation flip --img-views 2 --source product --num-classes 15
python source_features.py --dataset officehome --img-augmentation flip --img-views 2 --source realworld --num-classes 15
#
python source_features.py --dataset domainnet --img-augmentation flip --img-views 2 --source painting --num-classes 200
python source_features.py --dataset domainnet --img-augmentation flip --img-views 2 --source real --num-classes 200
python source_features.py --dataset domainnet --img-augmentation flip --img-views 2 --source sketch --num-classes 200

python source_features.py --dataset visda --img-augmentation flip --img-views 2 --source train --num-classes 9