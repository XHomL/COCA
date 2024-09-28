#python coca.py --modality uni_modal --dataset officehome --source art --target clipart --task OPDA
#python coca.py --modality uni_modal --dataset officehome --source art --target product --task OPDA
#python coca.py --modality uni_modal --dataset officehome --source art --target realworld --task OPDA
#python coca.py --modality uni_modal --dataset officehome --source clipart --target art --task OPDA
#python coca.py --modality uni_modal --dataset officehome --source clipart --target product --task OPDA
#python coca.py --modality uni_modal --dataset officehome --source clipart --target realworld --task OPDA
#python coca.py --modality uni_modal --dataset officehome --source product --target art --task OPDA
#python coca.py --modality uni_modal --dataset officehome --source product --target clipart --task OPDA
#python coca.py --modality uni_modal --dataset officehome --source product --target realworld --task OPDA
#python coca.py --modality uni_modal --dataset officehome --source realworld --target art --task OPDA
#python coca.py --modality uni_modal --dataset officehome --source realworld --target clipart --task OPDA
#python coca.py --modality uni_modal --dataset officehome --source realworld --target product --task OPDA
#
#python coca.py --modality uni_modal --dataset visda --source train --target validation --task OPDA

python coca.py --modality uni_modal --dataset domainnet --source painting --target real --task OPDA
python coca.py --modality uni_modal --dataset domainnet --source painting --target sketch --task OPDA
python coca.py --modality uni_modal --dataset domainnet --source real --target painting --task OPDA
python coca.py --modality uni_modal --dataset domainnet --source real --target sketch --task OPDA
python coca.py --modality uni_modal --dataset domainnet --source sketch --target painting --task OPDA
python coca.py --modality uni_modal --dataset domainnet --source sketch --target real --task OPDA