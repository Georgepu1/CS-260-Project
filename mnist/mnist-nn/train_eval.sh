# Lowest model complexity
python3 train_nn.py --trial 5 --width 100 --noise 0.0
python3 train_nn.py --trial 5 --width 100 --noise 0.2
python3 train_nn.py --trial 5 --width 100 --noise 0.4
python3 train_nn.py --trial 5 --width 100 --noise 0.6

# Next lowest model complexity
python3 train_nn.py --trial 5 --width 200 --noise 0.0
python3 train_nn.py --trial 5 --width 200 --noise 0.2
python3 train_nn.py --trial 5 --width 200 --noise 0.4
python3 train_nn.py --trial 5 --width 200 --noise 0.6

# Third lowest model complexity
python3 train_nn.py --trial 5 --width 400 --noise 0.0
python3 train_nn.py --trial 5 --width 400 --noise 0.2
python3 train_nn.py --trial 5 --width 400 --noise 0.4
python3 train_nn.py --trial 5 --width 400 --noise 0.6

# Second largest model complexity
python3 train_nn.py --trial 5 --width 800 --noise 0.0
python3 train_nn.py --trial 5 --width 800 --noise 0.2
python3 train_nn.py --trial 5 --width 800 --noise 0.4
python3 train_nn.py --trial 5 --width 800 --noise 0.6

# Largest model complexity
python3 train_nn.py --trial 5 --width 1600 --noise 0.0
python3 train_nn.py --trial 5 --width 1600 --noise 0.2
python3 train_nn.py --trial 5 --width 1600 --noise 0.4
python3 train_nn.py --trial 5 --width 1600 --noise 0.6