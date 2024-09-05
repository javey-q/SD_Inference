from collections import OrderedDict
from copy import deepcopy
import numpy as np
import onnx
import os
import tempfile
from onnx import shape_inference
import onnx_graphsurgeon as gs
from polygraphy.backend.onnx.loader import fold_constants

class Optimizer():
    def __init__(
        self,
        onnx_graph,
        verbose=False
    ):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix=''):
        if self.verbose:
            print(f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs")

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        # 常量合并
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        # 形状推断
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            temp_dir = tempfile.TemporaryDirectory().name
            os.makedirs(temp_dir, exist_ok=True)
            onnx_orig_path = os.path.join(temp_dir, 'model.onnx')
            onnx_inferred_path = os.path.join(temp_dir, 'inferred.onnx')
            onnx.save_model(onnx_graph,
                onnx_orig_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                convert_attribute=False)
            onnx.shape_inference.infer_shapes_path(onnx_orig_path, onnx_inferred_path)
            onnx_graph = onnx.load(onnx_inferred_path)
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def remove_casts(self):
        nRemoveCastNode = 0
        for node in self.graph.nodes:
            # Remove Cast nodes before qkv gemm
            if node.op in ["Add", "Transpose"] and len(node.outputs[0].outputs) == 3 and node.o().op == "Cast" and node.o(1).op == "Cast" and node.o(2).op == "Cast":
                for i in range(len(node.outputs[0].outputs)):
                    matMulNode = node.o(i, 0).o()
                    matMulNode.inputs[0] = node.outputs[0]
                    nRemoveCastNode += 1

            # Remove double cast nodes after Softmax Node
            if node.op == "Softmax" and node.o().op == "Cast" and node.o().o().op == "Cast":
                node.o().o().o().inputs[0] = node.outputs[0]
                nRemoveCastNode += 1

        self.cleanup()
        return nRemoveCastNode

    def remove_parallel_swish(self):
        mRemoveSwishNode = 0
        for node in self.graph.nodes:
            if node.op == "Gemm" and len(node.outputs[0].outputs) > 6:
                swishOutputTensor = None
                for nextNode in node.outputs[0].outputs:
                    if nextNode.op == "Mul":
                        if swishOutputTensor is None:
                            swishOutputTensor = nextNode.outputs[0]
                        else:
                            nextGemmNode = nextNode.o(0)
                            assert nextGemmNode.op == "Gemm", "Unexpected node type for nextGemmNode {}".format(nextGemmNode.name)
                            nextGemmNode.inputs = [swishOutputTensor, nextGemmNode.inputs[1], nextGemmNode.inputs[2]]
                            nextNode.outputs.clear()
                            mRemoveSwishNode += 1

        self.cleanup()
        return mRemoveSwishNode

    def resize_fix(self):
        '''
        This function loops through the graph looking for Resize nodes that uses scales for resize (has 3 inputs).
        It substitutes found Resize with Resize that takes the size of the output tensor instead of scales.
        It adds Shape->Slice->Concat
                Shape->Slice----^     subgraph to the graph to extract the shape of the output tensor.
        This fix is required for the dynamic shape support.
        '''
        mResizeNodes = 0
        for node in self.graph.nodes:
            if node.op == "Resize" and len(node.inputs) == 3:
                name = node.name + "/"
                
                add_node = node.o().o().i(1)
                div_node = node.i()
                
                shape_hw_out = gs.Variable(name=name + "shape_hw_out", dtype=np.int64, shape=[4])
                shape_hw = gs.Node(op="Shape", name=name+"shape_hw", inputs=[add_node.outputs[0]], outputs=[shape_hw_out])

                const_zero = gs.Constant(name=name + "const_zero", values=np.array([0], dtype=np.int64))
                const_two = gs.Constant(name=name + "const_two", values=np.array([2], dtype=np.int64))
                const_four = gs.Constant(name=name + "const_four", values=np.array([4], dtype=np.int64))

                slice_hw_out = gs.Variable(name=name + "slice_hw_out", dtype=np.int64, shape=[2])
                slice_hw = gs.Node(op="Slice", name=name+"slice_hw", inputs=[shape_hw_out, const_two, const_four, const_zero], outputs=[slice_hw_out])

                shape_bc_out = gs.Variable(name=name + "shape_bc_out", dtype=np.int64, shape=[2])
                shape_bc = gs.Node(op="Shape", name=name+"shape_bc", inputs=[div_node.outputs[0]], outputs=[shape_bc_out])

                slice_bc_out = gs.Variable(name=name + "slice_bc_out", dtype=np.int64, shape=[2])
                slice_bc = gs.Node(op="Slice", name=name+"slice_bc", inputs=[shape_bc_out, const_zero, const_two, const_zero], outputs=[slice_bc_out])

                concat_bchw_out = gs.Variable(name=name + "concat_bchw_out", dtype=np.int64, shape=[4])
                concat_bchw = gs.Node(op="Concat", name=name+"concat_bchw", attrs={"axis": 0}, inputs=[slice_bc_out, slice_hw_out], outputs=[concat_bchw_out])

                none_var = gs.Variable.empty()

                resize_bchw = gs.Node(op="Resize", name=name+"resize_bchw", attrs=node.attrs, inputs=[node.inputs[0], none_var, none_var, concat_bchw_out], outputs=[node.outputs[0]])

                self.graph.nodes.extend([shape_hw, slice_hw, shape_bc, slice_bc, concat_bchw, resize_bchw])

                node.inputs = []
                node.outputs = []

                mResizeNodes += 1

        self.cleanup()
        return mResizeNodes


    def adjustAddNode(self):
        nAdjustAddNode = 0
        for node in self.graph.nodes:
            # Change the bias const to the second input to allow Gemm+BiasAdd fusion in TRT.
            if node.op in ["Add"] and isinstance(node.inputs[0], gs.ir.tensor.Constant):
                tensor = node.inputs[1]
                bias = node.inputs[0]
                node.inputs = [tensor, bias]
                nAdjustAddNode += 1

        self.cleanup()
        return nAdjustAddNode

    def decompose_instancenorms(self):
        nRemoveInstanceNorm = 0
        for node in self.graph.nodes:
            if node.op == "InstanceNormalization":
                name = node.name + "/"
                input_tensor = node.inputs[0]
                output_tensor = node.outputs[0]
                mean_out = gs.Variable(name=name + "mean_out")
                mean_node = gs.Node(op="ReduceMean", name=name + "mean_node", attrs={"axes": [-1]}, inputs=[input_tensor], outputs=[mean_out])
                sub_out = gs.Variable(name=name + "sub_out")
                sub_node = gs.Node(op="Sub", name=name + "sub_node", attrs={}, inputs=[input_tensor, mean_out], outputs=[sub_out])
                pow_out = gs.Variable(name=name + "pow_out")
                pow_const = gs.Constant(name=name + "pow_const", values=np.array([2.0], dtype=np.float32))
                pow_node = gs.Node(op="Pow", name=name + "pow_node", attrs={}, inputs=[sub_out, pow_const], outputs=[pow_out])
                mean2_out = gs.Variable(name=name + "mean2_out")
                mean2_node = gs.Node(op="ReduceMean", name=name + "mean2_node", attrs={"axes": [-1]}, inputs=[pow_out], outputs=[mean2_out])
                epsilon_out = gs.Variable(name=name + "epsilon_out")
                epsilon_const = gs.Constant(name=name + "epsilon_const", values=np.array([node.attrs["epsilon"]], dtype=np.float32))
                epsilon_node = gs.Node(op="Add", name=name + "epsilon_node", attrs={}, inputs=[mean2_out, epsilon_const], outputs=[epsilon_out])
                sqrt_out = gs.Variable(name=name + "sqrt_out")
                sqrt_node = gs.Node(op="Sqrt", name=name + "sqrt_node", attrs={}, inputs=[epsilon_out], outputs=[sqrt_out])
                div_out = gs.Variable(name=name + "div_out")
                div_node = gs.Node(op="Div", name=name + "div_node", attrs={}, inputs=[sub_out, sqrt_out], outputs=[div_out])
                constantScale = gs.Constant("InstanceNormScaleV-" + str(nRemoveInstanceNorm), np.ascontiguousarray(node.inputs[1].inputs[0].attrs["value"].values.reshape(1, 32, 1)))
                constantBias = gs.Constant("InstanceBiasV-" + str(nRemoveInstanceNorm), np.ascontiguousarray(node.inputs[2].inputs[0].attrs["value"].values.reshape(1, 32, 1)))
                mul_out = gs.Variable(name=name + "mul_out")
                mul_node = gs.Node(op="Mul", name=name + "mul_node", attrs={}, inputs=[div_out, constantScale], outputs=[mul_out])
                add_node = gs.Node(op="Add", name=name + "add_node", attrs={}, inputs=[mul_out, constantBias], outputs=[output_tensor])
                self.graph.nodes.extend([mean_node, sub_node, pow_node, mean2_node, epsilon_node, sqrt_node, div_node, mul_node, add_node])
                node.inputs = []
                node.outputs = []
                nRemoveInstanceNorm += 1

        self.cleanup()
        return nRemoveInstanceNorm

    def insert_groupnorm_plugin(self):
        nGroupNormPlugin = 0
        for node in self.graph.nodes:
            if (
                node.op == "Reshape"
                and node.o().op == "InstanceNormalization"
                and node.o().o().op == "Reshape"
                and node.o().o().o().op == "Mul"
                and node.o().o().o().o().op == "Add"
            ):

                last_node = node.o().o().o().o()

                instance_norm = node.o()
                instance_norm_scale = instance_norm.inputs[1]
                instance_norm_bias = instance_norm.inputs[2]
                epsilon = instance_norm.attrs["epsilon"]
                mul_node = node.o().o().o()
                add_node = node.o().o().o().o()

                scale = np.ascontiguousarray(np.array(deepcopy(instance_norm_scale.values.tolist()), dtype=np.float32))
                bias = np.ascontiguousarray(np.array(deepcopy(instance_norm_bias.values.tolist()), dtype=np.float32))
                gamma = np.ascontiguousarray(np.array(deepcopy(mul_node.inputs[1].values.tolist()), dtype=np.float32))
                beta = np.ascontiguousarray(np.array(deepcopy(add_node.inputs[1].values.tolist()), dtype=np.float32))
                

                with_swish = (
                    True
                    if node.o().o().o().o().o().op == "Sigmoid"
                    and node.o().o().o().o().o().o().op == "Mul"
                    else False
                )
                if with_swish:
                    last_node = node.o().o().o().o().o().o()

                constant_gamma = gs.Constant("gamma_{}".format(nGroupNormPlugin), gamma.reshape(-1))
                constant_beta = gs.Constant("beta_{}".format(nGroupNormPlugin), beta.reshape(-1))
                x = node.inputs[0]
                group_norm_v = gs.Variable("group_norm_{}".format(nGroupNormPlugin), np.dtype(np.float32), x.shape)
                group_norm = gs.Node(
                    "GroupNorm",
                    "GroupNorm_{}".format(nGroupNormPlugin),
                    attrs={"epsilon": epsilon, "bSwish": with_swish},
                    inputs=[x, constant_gamma, constant_beta],
                    outputs=[group_norm_v],
                )
                nGroupNormPlugin += 1
                for n in self.graph.nodes:
                    if last_node.outputs[0] in n.inputs:
                        index = n.inputs.index(last_node.outputs[0])
                        n.inputs[index] = group_norm.outputs[0]
                last_node.outputs = []
                self.graph.nodes.append(group_norm)

        self.cleanup()
        return nGroupNormPlugin

    def insert_layernorm_plugin(self):
        nLayerNormPlugin = 0
        for node in self.graph.nodes:
            if node.op == 'ReduceMean' and \
                node.o().op == 'Sub' and node.o().inputs[0] == node.inputs[0] and \
                node.o().o(0).op =='Pow' and node.o().o(1).op =='Div' and \
                node.o().o(0).o().op == 'ReduceMean' and \
                node.o().o(0).o().o().op == 'Add' and \
                node.o().o(0).o().o().o().op == 'Sqrt' and \
                node.o().o(0).o().o().o().o().op == 'Div' and node.o().o(0).o().o().o().o() == node.o().o(1) and \
                node.o().o(0).o().o().o().o().o().op == 'Mul' and \
                node.o().o(0).o().o().o().o().o().o().op == 'Add' and \
                len(node.o().o(0).o().o().o().o().o().inputs[1].values.shape) == 1:

                if node.i().op == "Add":
                    inputTensor = node.inputs[0]  # CLIP
                else:
                    inputTensor = node.i().inputs[0]  # UNet and VAE

                # The first axis to normalize from can be inferred from the size of the `axes`
                # parameter of (any of) the `ReduceMean` node(s)
                reduceMeanNode = node.o().o(0).o()
                assert reduceMeanNode.op == "ReduceMean"
                firstNormAxis = -1 * np.size(np.array(reduceMeanNode.attrs["axes"]))

                gammaNode = node.o().o().o().o().o().o().o()
                index = [type(i) == gs.ir.tensor.Constant for i in gammaNode.inputs].index(True)
                gamma = np.array(deepcopy(gammaNode.inputs[index].values.tolist()), dtype=np.float32)
                constantGamma = gs.Constant("LayerNormGamma-" + str(nLayerNormPlugin), np.ascontiguousarray(gamma.reshape(-1)))  # MUST use np.ascontiguousarray, or TRT will regard the shape of this Constant as (0) !!!

                betaNode = gammaNode.o()
                index = [type(i) == gs.ir.tensor.Constant for i in betaNode.inputs].index(True)
                beta = np.array(deepcopy(betaNode.inputs[index].values.tolist()), dtype=np.float32)
                constantBeta = gs.Constant("LayerNormBeta-" + str(nLayerNormPlugin), np.ascontiguousarray(beta.reshape(-1)))

                inputList = [inputTensor, constantGamma, constantBeta]
                layerNormV = gs.Variable("LayerNormV-" + str(nLayerNormPlugin), np.dtype(np.float32), inputTensor.shape)
                layerNormN = gs.Node("LayerNorm", "LayerNormN-" + str(nLayerNormPlugin), inputs=inputList, attrs=OrderedDict([('epsilon', 1.e-5), ('axis', firstNormAxis)]), outputs=[layerNormV])
                self.graph.nodes.append(layerNormN)
                nLayerNormPlugin += 1

                if betaNode.outputs[0] in self.graph.outputs:
                    index = self.graph.outputs.index(betaNode.outputs[0])
                    self.graph.outputs[index] = layerNormV
                else:
                    if betaNode.o().op == "Cast":
                        lastNode = betaNode.o()
                    else:
                        lastNode = betaNode
                    for subNode in self.graph.nodes:
                        if lastNode.outputs[0] in subNode.inputs:
                            index = subNode.inputs.index(lastNode.outputs[0])
                            subNode.inputs[index] = layerNormV
                    lastNode.outputs = []

        self.cleanup()
        return nLayerNormPlugin

    def insert_splitgelu_plugin(self):
        nSplitGeLUPlugin = 0
        for node in self.graph.nodes:
            if node.op == "Erf":
                inputTensor = node.i().i().i().outputs[0]
                lastNode = node.o().o().o().o()
                outputShape = inputTensor.shape
                outputShape[2] = outputShape[2] // 2

                splitGeLUV = gs.Variable("splitGeLUV-" + str(nSplitGeLUPlugin), np.dtype(np.float32), outputShape)
                splitGeLUN = gs.Node("SplitGeLU", "splitGeLUN-" + str(nSplitGeLUPlugin), inputs=[inputTensor], outputs=[splitGeLUV])
                self.graph.nodes.append(splitGeLUN)

                for subNode in self.graph.nodes:
                    if lastNode.outputs[0] in subNode.inputs:
                        index = subNode.inputs.index(lastNode.outputs[0])
                        subNode.inputs[index] = splitGeLUV
                lastNode.outputs = []
                nSplitGeLUPlugin += 1

        self.cleanup()
        return nSplitGeLUPlugin

    def insert_seq2spatial_plugin(self):
        nSeqLen2SpatialPlugin = 0
        for node in self.graph.nodes:
            if node.op == "Transpose" and node.o().op == "Conv":
                transposeNode = node
                reshapeNode = node.i()
                assert reshapeNode.op == "Reshape", "Unexpected node type for reshapeNode {}".format(reshapeNode.name)
                residualNode = reshapeNode.i(0)
                assert residualNode.op == "Add", "Unexpected node type for residualNode {}".format(residualNode.name)
                biasNode = residualNode.i(0)
                assert biasNode.op == "Add", "Unexpected node type for biasNode {}".format(biasNode.name)
                biasIndex = [type(i) == gs.ir.tensor.Constant for i in biasNode.inputs].index(True)
                bias = np.array(deepcopy(biasNode.inputs[biasIndex].values.tolist()), dtype=np.float32)
                biasInput = gs.Constant("AddAddSeqLen2SpatialBias-" + str(nSeqLen2SpatialPlugin), np.ascontiguousarray(bias.reshape(-1)))
                inputIndex = 1 - biasIndex
                inputTensor = biasNode.inputs[inputIndex]
                residualInput = residualNode.inputs[1]
                outputTensor = transposeNode.outputs[0]
                outputShapeTensor = transposeNode.i().i().i(1).i(1).i(1).i().inputs[0]
                seqLen2SpatialNode = gs.Node("SeqLen2Spatial", "AddAddSeqLen2Spatial-" + str(nSeqLen2SpatialPlugin),
                    inputs=[inputTensor, biasInput, residualInput, outputShapeTensor], outputs=[outputTensor])
                self.graph.nodes.append(seqLen2SpatialNode)
                biasNode.inputs.clear()
                transposeNode.outputs.clear()
                nSeqLen2SpatialPlugin += 1

        self.cleanup()
        return nSeqLen2SpatialPlugin

    def fuse_kv(self, node_k, node_v, fused_kv_idx, heads, num_dynamic=0):
        # Get weights of K
        weights_k = node_k.inputs[1].values
        # Get weights of V
        weights_v = node_v.inputs[1].values
        # Input number of channels to K and V
        C = weights_k.shape[0]
        # Number of heads
        H = heads
        # Dimension per head
        D = weights_k.shape[1] // H

        # Concat and interleave weights such that the output of fused KV GEMM has [b, s_kv, h, 2, d] shape
        weights_kv = np.dstack([weights_k.reshape(C, H, D), weights_v.reshape(C, H, D)]).reshape(C, 2 * H * D)

        # K and V have the same input
        input_tensor = node_k.inputs[0]
        # K and V must have the same output which we feed into fmha plugin
        output_tensor_k = node_k.outputs[0]
        # Create tensor
        constant_weights_kv = gs.Constant("Weights_KV_{}".format(fused_kv_idx), np.ascontiguousarray(weights_kv))

        # Create fused KV node
        fused_kv_node = gs.Node(op="MatMul", name="MatMul_KV_{}".format(fused_kv_idx), inputs=[input_tensor, constant_weights_kv], outputs=[output_tensor_k])
        self.graph.nodes.append(fused_kv_node)

        # Connect the output of fused node to the inputs of the nodes after K and V
        node_v.o(num_dynamic).inputs[0] = output_tensor_k
        node_k.o(num_dynamic).inputs[0] = output_tensor_k
        for i in range(0,num_dynamic):
            node_v.o().inputs.clear()
            node_k.o().inputs.clear()

        # Clear inputs and outputs of K and V to ge these nodes cleared
        node_k.outputs.clear()
        node_v.outputs.clear()
        node_k.inputs.clear()
        node_v.inputs.clear()

        self.cleanup()
        return fused_kv_node

    def insert_fmhca(self, node_q, node_kv, final_tranpose, mhca_idx, heads, num_dynamic=0):
        # Get inputs and outputs for the fMHCA plugin
        # We take an output of reshape that follows the Q GEMM
        output_q = node_q.o(num_dynamic).o().inputs[0]
        output_kv = node_kv.o().inputs[0]
        output_final_tranpose = final_tranpose.outputs[0]

        # Clear the inputs of the nodes that follow the Q and KV GEMM
        # to delete these subgraphs (it will be substituted by fMHCA plugin)
        node_kv.outputs[0].outputs[0].inputs.clear()
        node_kv.outputs[0].outputs[0].inputs.clear()
        node_q.o(num_dynamic).o().inputs.clear()
        for i in range(0,num_dynamic):
            node_q.o(i).o().o(1).inputs.clear()

        weights_kv = node_kv.inputs[1].values
        dims_per_head = weights_kv.shape[1] // (heads * 2)

        # Reshape dims
        shape = gs.Constant("Shape_KV_{}".format(mhca_idx), np.ascontiguousarray(np.array([0, 0, heads, 2, dims_per_head], dtype=np.int64)))

        # Reshape output tensor
        output_reshape = gs.Variable("ReshapeKV_{}".format(mhca_idx), np.dtype(np.float16), None)
        # Create fMHA plugin
        reshape = gs.Node(op="Reshape", name="Reshape_{}".format(mhca_idx), inputs=[output_kv, shape], outputs=[output_reshape])
        # Insert node
        self.graph.nodes.append(reshape)

        # Create fMHCA plugin
        fmhca = gs.Node(op="fMHCA", name="fMHCA_{}".format(mhca_idx), inputs=[output_q, output_reshape], outputs=[output_final_tranpose])
        # Insert node
        self.graph.nodes.append(fmhca)

        # Connect input of fMHCA to output of Q GEMM
        node_q.o(num_dynamic).outputs[0] = output_q

        if num_dynamic > 0:
            reshape2_input1_out = gs.Variable("Reshape2_fmhca{}_out".format(mhca_idx), np.dtype(np.int64), None)
            reshape2_input1_shape = gs.Node("Shape", "Reshape2_fmhca{}_shape".format(mhca_idx), inputs=[node_q.inputs[0]], outputs=[reshape2_input1_out])
            self.graph.nodes.append(reshape2_input1_shape)
            final_tranpose.o().inputs[1] = reshape2_input1_out

        # Clear outputs of transpose to get this subgraph cleared
        final_tranpose.outputs.clear()

        self.cleanup()

    def fuse_qkv(self, node_q, node_k, node_v, fused_qkv_idx, heads, num_dynamic=0):
        # Get weights of Q
        weights_q = node_q.inputs[1].values
        # Get weights of K
        weights_k = node_k.inputs[1].values
        # Get weights of V
        weights_v = node_v.inputs[1].values

        # Input number of channels to Q, K and V
        C = weights_k.shape[0]
        # Number of heads
        H = heads
        # Hidden dimension per head
        D = weights_k.shape[1] // H

        # Concat and interleave weights such that the output of fused QKV GEMM has [b, s, h, 3, d] shape
        weights_qkv = np.dstack([weights_q.reshape(C, H, D), weights_k.reshape(C, H, D), weights_v.reshape(C, H, D)]).reshape(C, 3 * H * D)

        input_tensor = node_k.inputs[0]  # K and V have the same input
        # Q, K and V must have the same output which we feed into fmha plugin
        output_tensor_k = node_k.outputs[0]
        # Concat and interleave weights such that the output of fused QKV GEMM has [b, s, h, 3, d] shape
        constant_weights_qkv = gs.Constant("Weights_QKV_{}".format(fused_qkv_idx), np.ascontiguousarray(weights_qkv))

        # Created a fused node
        fused_qkv_node = gs.Node(op="MatMul", name="MatMul_QKV_{}".format(fused_qkv_idx), inputs=[input_tensor, constant_weights_qkv], outputs=[output_tensor_k])
        self.graph.nodes.append(fused_qkv_node)

        # Connect the output of the fused node to the inputs of the nodes after Q, K and V
        node_q.o(num_dynamic).inputs[0] = output_tensor_k
        node_k.o(num_dynamic).inputs[0] = output_tensor_k
        node_v.o(num_dynamic).inputs[0] = output_tensor_k
        for i in range(0,num_dynamic):
            node_q.o().inputs.clear()
            node_k.o().inputs.clear()
            node_v.o().inputs.clear()

        # Clear inputs and outputs of Q, K and V to ge these nodes cleared
        node_q.outputs.clear()
        node_k.outputs.clear()
        node_v.outputs.clear()

        node_q.inputs.clear()
        node_k.inputs.clear()
        node_v.inputs.clear()

        self.cleanup()
        return fused_qkv_node

    def insert_fmha(self, node_qkv, final_tranpose, mha_idx, heads, num_dynamic=0):
        # Get inputs and outputs for the fMHA plugin
        output_qkv = node_qkv.o().inputs[0]
        output_final_tranpose = final_tranpose.outputs[0]

        # Clear the inputs of the nodes that follow the QKV GEMM
        # to delete these subgraphs (it will be substituted by fMHA plugin)
        node_qkv.outputs[0].outputs[2].inputs.clear()
        node_qkv.outputs[0].outputs[1].inputs.clear()
        node_qkv.outputs[0].outputs[0].inputs.clear()

        weights_qkv = node_qkv.inputs[1].values
        dims_per_head = weights_qkv.shape[1] // (heads * 3)

        # Reshape dims
        shape = gs.Constant("Shape_QKV_{}".format(mha_idx), np.ascontiguousarray(np.array([0, 0, heads, 3, dims_per_head], dtype=np.int64)))

        # Reshape output tensor
        output_shape = gs.Variable("ReshapeQKV_{}".format(mha_idx), np.dtype(np.float16), None)
        # Create fMHA plugin
        reshape = gs.Node(op="Reshape", name="Reshape_{}".format(mha_idx), inputs=[output_qkv, shape], outputs=[output_shape])
        # Insert node
        self.graph.nodes.append(reshape)

        # Create fMHA plugin
        fmha = gs.Node(op="fMHA_V2", name="fMHA_{}".format(mha_idx), inputs=[output_shape], outputs=[output_final_tranpose])
        # Insert node
        self.graph.nodes.append(fmha)

        if num_dynamic > 0:
            reshape2_input1_out = gs.Variable("Reshape2_{}_out".format(mha_idx), np.dtype(np.int64), None)
            reshape2_input1_shape = gs.Node("Shape", "Reshape2_{}_shape".format(mha_idx), inputs=[node_qkv.inputs[0]], outputs=[reshape2_input1_out])
            self.graph.nodes.append(reshape2_input1_shape)
            final_tranpose.o().inputs[1] = reshape2_input1_out

        # Clear outputs of transpose to get this subgraph cleared
        final_tranpose.outputs.clear()

        self.cleanup()

    def mha_mhca_detected(self, node, mha):
        # Go from V GEMM down to the S*V MatMul and all way up to K GEMM
        # If we are looking for MHCA inputs of two matmuls (K and V) must be equal.
        # If we are looking for MHA inputs (K and V) must be not equal.
        if node.op == "MatMul" and len(node.outputs) == 1 and \
            ((mha and len(node.inputs[0].inputs) > 0  and node.i().op == "Add") or \
            (not mha and len(node.inputs[0].inputs) == 0)):

            if node.o().op == 'Shape':
                if node.o(1).op == 'Shape':
                    num_dynamic_kv = 3 if node.o(2).op == 'Shape' else 2
                else:
                    num_dynamic_kv = 1
                # For Cross-Attention, if batch axis is dynamic (in QKV), assume H*W (in Q) is dynamic as well
                num_dynamic_q = num_dynamic_kv if mha else num_dynamic_kv + 1
            else:
                num_dynamic_kv = 0
                num_dynamic_q = 0

            o = node.o(num_dynamic_kv)
            if o.op == "Reshape" and \
                o.o().op == "Transpose" and \
                o.o().o().op == "Reshape" and \
                o.o().o().o().op == "MatMul" and \
                o.o().o().o().i(0).op == "Softmax" and \
                o.o().o().o().i(1).op == "Reshape" and \
                o.o().o().o().i(0).i().op == "Mul" and \
                o.o().o().o().i(0).i().i().op == "MatMul" and \
                o.o().o().o().i(0).i().i().i(0).op == "Reshape" and \
                o.o().o().o().i(0).i().i().i(1).op == "Transpose" and \
                o.o().o().o().i(0).i().i().i(1).i().op == "Reshape" and \
                o.o().o().o().i(0).i().i().i(1).i().i().op == "Transpose" and \
                o.o().o().o().i(0).i().i().i(1).i().i().i().op == "Reshape" and \
                o.o().o().o().i(0).i().i().i(1).i().i().i().i().op == "MatMul" and \
                node.name != o.o().o().o().i(0).i().i().i(1).i().i().i().i().name:
                # "len(node.outputs) == 1" to make sure we are not in the already fused node
                node_q = o.o().o().o().i(0).i().i().i(0).i().i().i()
                node_k = o.o().o().o().i(0).i().i().i(1).i().i().i().i()
                node_v = node
                final_tranpose = o.o().o().o().o(num_dynamic_q).o()
                # Sanity check to make sure that the graph looks like expected
                if node_q.op == "MatMul" and final_tranpose.op == "Transpose":
                    return True, num_dynamic_q, num_dynamic_kv, node_q, node_k, node_v, final_tranpose
        return False, 0, 0, None, None, None, None

    def fuse_kv_insert_fmhca(self, heads, mhca_index, sm):
        nodes = self.graph.nodes
        # Iterate over graph and search for MHCA pattern
        for idx, _ in enumerate(nodes):
            # fMHCA can't be at the 2 last layers of the network. It is a guard from OOB
            if idx + 1 > len(nodes) or idx + 2 > len(nodes):
                continue

            # Get anchor nodes for fusion and fMHCA plugin insertion if the MHCA is detected
            detected, num_dynamic_q, num_dynamic_kv, node_q, node_k, node_v, final_tranpose = \
                self.mha_mhca_detected(nodes[idx], mha=False)
            if detected:
                assert num_dynamic_q == 0 or num_dynamic_q == num_dynamic_kv + 1
                # Skip the FMHCA plugin for SM75 except for when the dim per head is 40.
                if sm == 75 and node_q.inputs[1].shape[1] // heads == 160:
                    continue
                # Fuse K and V GEMMS
                node_kv = self.fuse_kv(node_k, node_v, mhca_index, heads, num_dynamic_kv)
                # Insert fMHCA plugin
                self.insert_fmhca(node_q, node_kv, final_tranpose, mhca_index, heads, num_dynamic_q)
                return True
        return False

    def fuse_qkv_insert_fmha(self, heads, mha_index):
        nodes = self.graph.nodes
        # Iterate over graph and search for MHA pattern
        for idx, _ in enumerate(nodes):
            # fMHA can't be at the 2 last layers of the network. It is a guard from OOB
            if idx + 1 > len(nodes) or idx + 2 > len(nodes):
                continue

            # Get anchor nodes for fusion and fMHA plugin insertion if the MHA is detected
            detected, num_dynamic_q, num_dynamic_kv, node_q, node_k, node_v, final_tranpose = \
                self.mha_mhca_detected(nodes[idx], mha=True)
            if detected:
                assert num_dynamic_q == num_dynamic_kv
                # Fuse Q, K and V GEMMS
                node_qkv = self.fuse_qkv(node_q, node_k, node_v, mha_index, heads, num_dynamic_kv)
                # Insert fMHA plugin
                self.insert_fmha(node_qkv, final_tranpose, mha_index, heads, num_dynamic_kv)
                return True
        return False

    def insert_fmhca_plugin(self, num_heads, sm):
        mhca_index = 0
        while self.fuse_kv_insert_fmhca(num_heads, mhca_index, sm):
            mhca_index += 1
        return mhca_index

    def insert_fmha_plugin(self, num_heads):
        mha_index = 0
        while self.fuse_qkv_insert_fmha(num_heads, mha_index):
            mha_index += 1
        return mha_index