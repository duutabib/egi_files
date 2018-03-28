
#!/bin/bash
# short.sh: a short discovery job
echo $PWD
echo "Working hard..."
module load python
module load keras

python resnet_script.py

echo $PWD
sleep 20
echo "Science complete!"
