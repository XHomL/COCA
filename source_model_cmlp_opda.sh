python source_model.py --dataset officehome --img-augmentation flip --img-views 2 --source art --num-classes 15
python source_model.py --dataset officehome --img-augmentation flip --img-views 2 --source clipart --num-classes 15
python source_model.py --dataset officehome --img-augmentation flip --img-views 2 --source product --num-classes 15
python source_model.py --dataset officehome --img-augmentation flip --img-views 2 --source realworld --num-classes 15

python source_model.py --dataset domainnet --img-augmentation flip --img-views 2 --source painting --num-classes 200
python source_model.py --dataset domainnet --img-augmentation flip --img-views 2 --source real --num-classes 200
python source_model.py --dataset domainnet --img-augmentation flip --img-views 2 --source sketch --num-classes 200

python source_model.py --dataset visda --img-augmentation flip --img-views 2 --source train --num-classes 9