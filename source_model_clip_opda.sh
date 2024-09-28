python source_model.py --modality uni_modal --dataset officehome --source art --num-classes 15
python source_model.py --modality uni_modal --dataset officehome --source clipart --num-classes 15
python source_model.py --modality uni_modal --dataset officehome --source product --num-classes 15
python source_model.py --modality uni_modal --dataset officehome --source realworld --num-classes 15

python source_model.py --modality uni_modal --dataset domainnet --source painting --num-classes 200
python source_model.py --modality uni_modal --dataset domainnet --source real --num-classes 200
python source_model.py --modality uni_modal --dataset domainnet --source sketch --num-classes 200

python source_model.py --modality uni_modal --dataset visda --source train --num-classes 9