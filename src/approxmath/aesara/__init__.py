from .basic import ApproxLogOp, ApproxExpOp
from .trig import ApproxCosOp, ApproxSinOp
import aesara.tensor as tt
import aesara

# initialize and export ops
x_exp = tt.matrix()
exp = aesara.function([x_exp], ApproxExpOp()(x_exp))

x_log = tt.matrix()
log = aesara.function([x_log], ApproxLogOp()(x_log))

x_cos = tt.matrix()
cos = aesara.function([x_cos], ApproxCosOp()(x_cos))

x_sin = tt.matrix()
sin = aesara.function([x_sin], ApproxSinOp()(x_sin))
