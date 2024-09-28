python coca.py --dataset officehome --source art --target clipart --task OPDA
python coca.py --dataset officehome --source art --target product --task OPDA
python coca.py --dataset officehome --source art --target realworld --task OPDA
python coca.py --dataset officehome --source clipart --target art --task OPDA
python coca.py --dataset officehome --source clipart --target product --task OPDA
python coca.py --dataset officehome --source clipart --target realworld --task OPDA
python coca.py --dataset officehome --source product --target art --task OPDA
python coca.py --dataset officehome --source product --target clipart --task OPDA
python coca.py --dataset officehome --source product --target realworld --task OPDA
python coca.py --dataset officehome --source realworld --target art --task OPDA
python coca.py --dataset officehome --source realworld --target clipart --task OPDA
python coca.py --dataset officehome --source realworld --target product --task OPDA

python coca.py --dataset visda --source train --target validation --task OPDA

python coca.py --dataset domainnet --sourcepainting--target real --task OPDA
python coca.py --dataset domainnet --sourcepainting--target sketch --task OPDA
python coca.py --dataset domainnet --source real --targetpainting--task OPDA
python coca.py --dataset domainnet --source real --target sketch --task OPDA
python coca.py --dataset domainnet --source sketch --targetpainting--task OPDA
python coca.py --dataset domainnet --source sketch --target real --task OPDA