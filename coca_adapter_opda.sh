python coca.py --classifier-head adapter --modality uni_modal --dataset officehome --source art --target clipart --task OPDA
python coca.py --classifier-head adapter --modality uni_modal --dataset officehome --source art --target product --task OPDA
python coca.py --classifier-head adapter --modality uni_modal --dataset officehome --source art --target realworld --task OPDA
python coca.py --classifier-head adapter --modality uni_modal --dataset officehome --source clipart --target art --task OPDA
python coca.py --classifier-head adapter --modality uni_modal --dataset officehome --source clipart --target product --task OPDA
python coca.py --classifier-head adapter --modality uni_modal --dataset officehome --source clipart --target realworld --task OPDA
python coca.py --classifier-head adapter --modality uni_modal --dataset officehome --source product --target art --task OPDA
python coca.py --classifier-head adapter --modality uni_modal --dataset officehome --source product --target clipart --task OPDA
python coca.py --classifier-head adapter --modality uni_modal --dataset officehome --source product --target realworld --task OPDA
python coca.py --classifier-head adapter --modality uni_modal --dataset officehome --source realworld --target art --task OPDA
python coca.py --classifier-head adapter --modality uni_modal --dataset officehome --source realworld --target clipart --task OPDA
python coca.py --classifier-head adapter --modality uni_modal --dataset officehome --source realworld --target product --task OPDA

python coca.py --classifier-head adapter --modality uni_modal --dataset visda --source train --target validation --task OPDA

python coca.py --classifier-head adapter --modality uni_modal --dataset domainnet --sourcepainting--target real --task OPDA
python coca.py --classifier-head adapter --modality uni_modal --dataset domainnet --sourcepainting--target sketch --task OPDA
python coca.py --classifier-head adapter --modality uni_modal --dataset domainnet --source real --targetpainting--task OPDA
python coca.py --classifier-head adapter --modality uni_modal --dataset domainnet --source real --target sketch --task OPDA
python coca.py --classifier-head adapter --modality uni_modal --dataset domainnet --source sketch --targetpainting--task OPDA
python coca.py --classifier-head adapter --modality uni_modal --dataset domainnet --source sketch --target real --task OPDA