
from tensorboard import program
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', "logs"])
url = tb.launch()

while(True):
    pass

#os.system('python -m tensorflow.tensorboard --logdir='+tf_log_dir)

