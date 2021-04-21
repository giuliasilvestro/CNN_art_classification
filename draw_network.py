import sys
sys.path.append('../')
from pycore.tikzeng import *

space = 5
# defined your arch
arch = [
    to_cor(),
    to_head( '..' ),
    to_begin(),
    to_input('picture.jpg', width=6, height=6,  name="image", to="(0,0,0)"),
    # Layer 1
    to_Conv("conv1", 256, "", 32, offset=f"({space},0,0)", to="(emotion-east)", height=16, width=16, depth=40, caption="ReLU+BN"),
    to_connection( "image", "conv1"), 
    # Layer 2
    to_Conv("conv2", 128, "", 32, offset=f"({space},0,0)", to="(conv1-east)", height=16, width=16, depth=32, caption="ReLU"),
    to_connection( "conv1", "conv2"), 
    to_Pool("maxpool1", to="(conv2-east)", height=8, width=8, depth=32, caption="MaxP+BN"),
    # Layer 3
    to_Conv("conv3", 64, "", 16, offset=f"({space},0,0)", to="(maxpool1-east)", height=8, width=8, depth=28, caption="ReLU+BN"),
    to_connection("maxpool1", "conv3"), 
    to_Pool("maxpool2",   to="(conv3-east)", height=4, width=4, depth=28, caption="MaxP+BN"),
    # Layer 4
    to_Conv("conv4", 32, "", 8, offset=f"({space},0,0)", to="(maxpool2-east)", height=4, width=4, depth=40, caption="ReLU+AdaMaxP+BN" ),  
    to_connection("maxpool2", "conv4"),
    to_Pool("maxpool3",  to="(conv4-east)", height=2, width=2, depth=40,),   
    # Flatten
    to_Flatten("flatten", n_unit=64, offset=f"({space},0,0)", to='(maxpool3-east)',  height=0.4, width=.4, depth=28, caption="FC+ReLU+BN"),
    to_connection("maxpool3", "flatten"),
    ### Softmax
    to_SoftMax("softmax", 5 ,f"({space},0,0)", "(flatten-east)", caption="SoftMax", height=.3, width=.3, depth=16),
    to_connection("flatten", "softmax"),
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
