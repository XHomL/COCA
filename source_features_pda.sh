python source_features.py --dataset officehome --img-augmentation none --source art --num-classes 65
python source_features.py --dataset officehome --img-augmentation none --source clipart --num-classes 65
python source_features.py --dataset officehome --img-augmentation none --source product --num-classes 65
python source_features.py --dataset officehome --img-augmentation none --source realworld --num-classes 65

python source_features.py --dataset visda --img-augmentation none --source train --num-classes 12

python source_features.py --dataset officehome --img-augmentation flip --img-views 2 --source art --num-classes 65
python source_features.py --dataset officehome --img-augmentation flip --img-views 2 --source clipart --num-classes 65
python source_features.py --dataset officehome --img-augmentation flip --img-views 2 --source product --num-classes 65
python source_features.py --dataset officehome --img-augmentation flip --img-views 2 --source realworld --num-classes 65

python source_features.py --dataset visda --img-augmentation flip --img-views 2 --source train --num-classes 12
