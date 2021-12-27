import aesara
from aesara.graph.op import Op
from aesara.graph.basic import Apply
import aesara.tensor as tt
import approxmath.np as npa

class ApproxSinOp(Op):
    __props__ = ()

    itypes = [aesara.tensor.dmatrix]
    otypes = [aesara.tensor.dmatrix]

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = npa.sin(x)

    def infer_shape(self, fgraph, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        # d/dx (sin x) => cos x
        return [tt.cos(inputs[0]) * output_grads[0]]

    def R_op(self, inputs, eval_points):
        # R_op can receive None as eval_points.
        # That mean there is no diferientiable path through that input
        # If this imply that you cannot compute some outputs,
        # return None for those.
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

class ApproxCosOp(Op):
    __props__ = ()

    itypes = [aesara.tensor.dmatrix]
    otypes = [aesara.tensor.dmatrix]

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = npa.cos(x)

    def infer_shape(self, fgraph, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        # d/dx (cos x) => -sin x
        return [-1 * tt.sin(inputs[0]) * output_grads[0]]

    def R_op(self, inputs, eval_points):
        # R_op can receive None as eval_points.
        # That mean there is no diferientiable path through that input
        # If this imply that you cannot compute some outputs,
        # return None for those.
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
