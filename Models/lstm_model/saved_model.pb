??)
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??'
|
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_10/kernel
u
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel* 
_output_shapes
:
??*
dtype0
s
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_10/bias
l
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes	
:?*
dtype0
w
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_nameoutput/kernel
p
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes
:	?*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
layer0/lstm_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	P? **
shared_namelayer0/lstm_cell_1/kernel
?
-layer0/lstm_cell_1/kernel/Read/ReadVariableOpReadVariableOplayer0/lstm_cell_1/kernel*
_output_shapes
:	P? *
dtype0
?
#layer0/lstm_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?? *4
shared_name%#layer0/lstm_cell_1/recurrent_kernel
?
7layer0/lstm_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOp#layer0/lstm_cell_1/recurrent_kernel* 
_output_shapes
:
?? *
dtype0
?
layer0/lstm_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *(
shared_namelayer0/lstm_cell_1/bias
?
+layer0/lstm_cell_1/bias/Read/ReadVariableOpReadVariableOplayer0/lstm_cell_1/bias*
_output_shapes	
:? *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_10/kernel/m
?
*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_10/bias/m
z
(Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameAdam/output/kernel/m
~
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
_output_shapes
:	?*
dtype0
|
Adam/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/m
u
&Adam/output/bias/m/Read/ReadVariableOpReadVariableOpAdam/output/bias/m*
_output_shapes
:*
dtype0
?
 Adam/layer0/lstm_cell_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	P? *1
shared_name" Adam/layer0/lstm_cell_1/kernel/m
?
4Adam/layer0/lstm_cell_1/kernel/m/Read/ReadVariableOpReadVariableOp Adam/layer0/lstm_cell_1/kernel/m*
_output_shapes
:	P? *
dtype0
?
*Adam/layer0/lstm_cell_1/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?? *;
shared_name,*Adam/layer0/lstm_cell_1/recurrent_kernel/m
?
>Adam/layer0/lstm_cell_1/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/layer0/lstm_cell_1/recurrent_kernel/m* 
_output_shapes
:
?? *
dtype0
?
Adam/layer0/lstm_cell_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:? */
shared_name Adam/layer0/lstm_cell_1/bias/m
?
2Adam/layer0/lstm_cell_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer0/lstm_cell_1/bias/m*
_output_shapes	
:? *
dtype0
?
Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_10/kernel/v
?
*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_10/bias/v
z
(Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameAdam/output/kernel/v
~
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes
:	?*
dtype0
|
Adam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/v
u
&Adam/output/bias/v/Read/ReadVariableOpReadVariableOpAdam/output/bias/v*
_output_shapes
:*
dtype0
?
 Adam/layer0/lstm_cell_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	P? *1
shared_name" Adam/layer0/lstm_cell_1/kernel/v
?
4Adam/layer0/lstm_cell_1/kernel/v/Read/ReadVariableOpReadVariableOp Adam/layer0/lstm_cell_1/kernel/v*
_output_shapes
:	P? *
dtype0
?
*Adam/layer0/lstm_cell_1/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?? *;
shared_name,*Adam/layer0/lstm_cell_1/recurrent_kernel/v
?
>Adam/layer0/lstm_cell_1/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/layer0/lstm_cell_1/recurrent_kernel/v* 
_output_shapes
:
?? *
dtype0
?
Adam/layer0/lstm_cell_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:? */
shared_name Adam/layer0/lstm_cell_1/bias/v
?
2Adam/layer0/lstm_cell_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer0/lstm_cell_1/bias/v*
_output_shapes	
:? *
dtype0

NoOpNoOp
?+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?*
value?*B?* B?*
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
l

cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
?
iter

beta_1

beta_2
	decay
 learning_ratemMmNmOmP!mQ"mR#mSvTvUvVvW!vX"vY#vZ
1
!0
"1
#2
3
4
5
6
1
!0
"1
#2
3
4
5
6
 
?
$layer_metrics
%non_trainable_variables

&layers
	variables
trainable_variables
'metrics
(layer_regularization_losses
regularization_losses
 
~

!kernel
"recurrent_kernel
#bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
 

!0
"1
#2

!0
"1
#2
 
?
-layer_metrics
.non_trainable_variables

/layers
trainable_variables
	variables
0metrics

1states
2layer_regularization_losses
regularization_losses
[Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_10/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
3layer_metrics
4non_trainable_variables

5layers
trainable_variables
	variables
6metrics
7layer_regularization_losses
regularization_losses
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
8layer_metrics
9non_trainable_variables

:layers
trainable_variables
	variables
;metrics
<layer_regularization_losses
regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer0/lstm_cell_1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#layer0/lstm_cell_1/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElayer0/lstm_cell_1/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
2

=0
>1
 

!0
"1
#2

!0
"1
#2
 
?
?layer_metrics
@non_trainable_variables

Alayers
)trainable_variables
*	variables
Bmetrics
Clayer_regularization_losses
+regularization_losses
 
 


0
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Dtotal
	Ecount
F	variables
G	keras_api
D
	Htotal
	Icount
J
_fn_kwargs
K	variables
L	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

D0
E1

F	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

H0
I1

K	variables
~|
VARIABLE_VALUEAdam/dense_10/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/layer0/lstm_cell_1/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/layer0/lstm_cell_1/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/layer0/lstm_cell_1/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/layer0/lstm_cell_1/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/layer0/lstm_cell_1/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/layer0/lstm_cell_1/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_layer0_inputPlaceholder*+
_output_shapes
:?????????P*
dtype0* 
shape:?????????P
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_layer0_inputlayer0/lstm_cell_1/kernellayer0/lstm_cell_1/bias#layer0/lstm_cell_1/recurrent_kerneldense_10/kerneldense_10/biasoutput/kerneloutput/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_349025
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-layer0/lstm_cell_1/kernel/Read/ReadVariableOp7layer0/lstm_cell_1/recurrent_kernel/Read/ReadVariableOp+layer0/lstm_cell_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_10/kernel/m/Read/ReadVariableOp(Adam/dense_10/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp4Adam/layer0/lstm_cell_1/kernel/m/Read/ReadVariableOp>Adam/layer0/lstm_cell_1/recurrent_kernel/m/Read/ReadVariableOp2Adam/layer0/lstm_cell_1/bias/m/Read/ReadVariableOp*Adam/dense_10/kernel/v/Read/ReadVariableOp(Adam/dense_10/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOp4Adam/layer0/lstm_cell_1/kernel/v/Read/ReadVariableOp>Adam/layer0/lstm_cell_1/recurrent_kernel/v/Read/ReadVariableOp2Adam/layer0/lstm_cell_1/bias/v/Read/ReadVariableOpConst*+
Tin$
"2 	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_351468
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_10/kerneldense_10/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelayer0/lstm_cell_1/kernel#layer0/lstm_cell_1/recurrent_kernellayer0/lstm_cell_1/biastotalcounttotal_1count_1Adam/dense_10/kernel/mAdam/dense_10/bias/mAdam/output/kernel/mAdam/output/bias/m Adam/layer0/lstm_cell_1/kernel/m*Adam/layer0/lstm_cell_1/recurrent_kernel/mAdam/layer0/lstm_cell_1/bias/mAdam/dense_10/kernel/vAdam/dense_10/bias/vAdam/output/kernel/vAdam/output/bias/v Adam/layer0/lstm_cell_1/kernel/v*Adam/layer0/lstm_cell_1/recurrent_kernel/vAdam/layer0/lstm_cell_1/bias/v**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_351568??&
??
?
while_body_350572
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
1while_lstm_cell_1_split_readvariableop_resource_07
3while_lstm_cell_1_split_1_readvariableop_resource_0/
+while_lstm_cell_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
/while_lstm_cell_1_split_readvariableop_resource5
1while_lstm_cell_1_split_1_readvariableop_resource-
)while_lstm_cell_1_readvariableop_resource?? while/lstm_cell_1/ReadVariableOp?"while/lstm_cell_1/ReadVariableOp_1?"while/lstm_cell_1/ReadVariableOp_2?"while/lstm_cell_1/ReadVariableOp_3?&while/lstm_cell_1/split/ReadVariableOp?(while/lstm_cell_1/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
!while/lstm_cell_1/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/ones_like/Shape?
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell_1/ones_like/Const?
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/ones_like?
while/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2!
while/lstm_cell_1/dropout/Const?
while/lstm_cell_1/dropout/MulMul$while/lstm_cell_1/ones_like:output:0(while/lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/dropout/Mul?
while/lstm_cell_1/dropout/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_1/dropout/Shape?
6while/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2?ª28
6while/lstm_cell_1/dropout/random_uniform/RandomUniform?
(while/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2*
(while/lstm_cell_1/dropout/GreaterEqual/y?
&while/lstm_cell_1/dropout/GreaterEqualGreaterEqual?while/lstm_cell_1/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2(
&while/lstm_cell_1/dropout/GreaterEqual?
while/lstm_cell_1/dropout/CastCast*while/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2 
while/lstm_cell_1/dropout/Cast?
while/lstm_cell_1/dropout/Mul_1Mul!while/lstm_cell_1/dropout/Mul:z:0"while/lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????P2!
while/lstm_cell_1/dropout/Mul_1?
!while/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!while/lstm_cell_1/dropout_1/Const?
while/lstm_cell_1/dropout_1/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????P2!
while/lstm_cell_1/dropout_1/Mul?
!while/lstm_cell_1/dropout_1/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_1/Shape?
8while/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2?ɵ2:
8while/lstm_cell_1/dropout_1/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_1/dropout_1/GreaterEqual/y?
(while/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2*
(while/lstm_cell_1/dropout_1/GreaterEqual?
 while/lstm_cell_1/dropout_1/CastCast,while/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2"
 while/lstm_cell_1/dropout_1/Cast?
!while/lstm_cell_1/dropout_1/Mul_1Mul#while/lstm_cell_1/dropout_1/Mul:z:0$while/lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????P2#
!while/lstm_cell_1/dropout_1/Mul_1?
!while/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!while/lstm_cell_1/dropout_2/Const?
while/lstm_cell_1/dropout_2/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????P2!
while/lstm_cell_1/dropout_2/Mul?
!while/lstm_cell_1/dropout_2/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_2/Shape?
8while/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2?ŏ2:
8while/lstm_cell_1/dropout_2/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_1/dropout_2/GreaterEqual/y?
(while/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2*
(while/lstm_cell_1/dropout_2/GreaterEqual?
 while/lstm_cell_1/dropout_2/CastCast,while/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2"
 while/lstm_cell_1/dropout_2/Cast?
!while/lstm_cell_1/dropout_2/Mul_1Mul#while/lstm_cell_1/dropout_2/Mul:z:0$while/lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????P2#
!while/lstm_cell_1/dropout_2/Mul_1?
!while/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!while/lstm_cell_1/dropout_3/Const?
while/lstm_cell_1/dropout_3/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????P2!
while/lstm_cell_1/dropout_3/Mul?
!while/lstm_cell_1/dropout_3/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_3/Shape?
8while/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2䴻2:
8while/lstm_cell_1/dropout_3/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_1/dropout_3/GreaterEqual/y?
(while/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2*
(while/lstm_cell_1/dropout_3/GreaterEqual?
 while/lstm_cell_1/dropout_3/CastCast,while/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2"
 while/lstm_cell_1/dropout_3/Cast?
!while/lstm_cell_1/dropout_3/Mul_1Mul#while/lstm_cell_1/dropout_3/Mul:z:0$while/lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????P2#
!while/lstm_cell_1/dropout_3/Mul_1?
#while/lstm_cell_1/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_1/ones_like_1/Shape?
#while/lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#while/lstm_cell_1/ones_like_1/Const?
while/lstm_cell_1/ones_like_1Fill,while/lstm_cell_1/ones_like_1/Shape:output:0,while/lstm_cell_1/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/ones_like_1?
!while/lstm_cell_1/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!while/lstm_cell_1/dropout_4/Const?
while/lstm_cell_1/dropout_4/MulMul&while/lstm_cell_1/ones_like_1:output:0*while/lstm_cell_1/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell_1/dropout_4/Mul?
!while/lstm_cell_1/dropout_4/ShapeShape&while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_4/Shape?
8while/lstm_cell_1/dropout_4/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_1/dropout_4/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_1/dropout_4/GreaterEqual/y?
(while/lstm_cell_1/dropout_4/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_4/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2*
(while/lstm_cell_1/dropout_4/GreaterEqual?
 while/lstm_cell_1/dropout_4/CastCast,while/lstm_cell_1/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2"
 while/lstm_cell_1/dropout_4/Cast?
!while/lstm_cell_1/dropout_4/Mul_1Mul#while/lstm_cell_1/dropout_4/Mul:z:0$while/lstm_cell_1/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2#
!while/lstm_cell_1/dropout_4/Mul_1?
!while/lstm_cell_1/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!while/lstm_cell_1/dropout_5/Const?
while/lstm_cell_1/dropout_5/MulMul&while/lstm_cell_1/ones_like_1:output:0*while/lstm_cell_1/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell_1/dropout_5/Mul?
!while/lstm_cell_1/dropout_5/ShapeShape&while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_5/Shape?
8while/lstm_cell_1/dropout_5/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ۺ2:
8while/lstm_cell_1/dropout_5/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_1/dropout_5/GreaterEqual/y?
(while/lstm_cell_1/dropout_5/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_5/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2*
(while/lstm_cell_1/dropout_5/GreaterEqual?
 while/lstm_cell_1/dropout_5/CastCast,while/lstm_cell_1/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2"
 while/lstm_cell_1/dropout_5/Cast?
!while/lstm_cell_1/dropout_5/Mul_1Mul#while/lstm_cell_1/dropout_5/Mul:z:0$while/lstm_cell_1/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2#
!while/lstm_cell_1/dropout_5/Mul_1?
!while/lstm_cell_1/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!while/lstm_cell_1/dropout_6/Const?
while/lstm_cell_1/dropout_6/MulMul&while/lstm_cell_1/ones_like_1:output:0*while/lstm_cell_1/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell_1/dropout_6/Mul?
!while/lstm_cell_1/dropout_6/ShapeShape&while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_6/Shape?
8while/lstm_cell_1/dropout_6/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_1/dropout_6/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_1/dropout_6/GreaterEqual/y?
(while/lstm_cell_1/dropout_6/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_6/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2*
(while/lstm_cell_1/dropout_6/GreaterEqual?
 while/lstm_cell_1/dropout_6/CastCast,while/lstm_cell_1/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2"
 while/lstm_cell_1/dropout_6/Cast?
!while/lstm_cell_1/dropout_6/Mul_1Mul#while/lstm_cell_1/dropout_6/Mul:z:0$while/lstm_cell_1/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2#
!while/lstm_cell_1/dropout_6/Mul_1?
!while/lstm_cell_1/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!while/lstm_cell_1/dropout_7/Const?
while/lstm_cell_1/dropout_7/MulMul&while/lstm_cell_1/ones_like_1:output:0*while/lstm_cell_1/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell_1/dropout_7/Mul?
!while/lstm_cell_1/dropout_7/ShapeShape&while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_7/Shape?
8while/lstm_cell_1/dropout_7/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??2:
8while/lstm_cell_1/dropout_7/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_1/dropout_7/GreaterEqual/y?
(while/lstm_cell_1/dropout_7/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_7/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2*
(while/lstm_cell_1/dropout_7/GreaterEqual?
 while/lstm_cell_1/dropout_7/CastCast,while/lstm_cell_1/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2"
 while/lstm_cell_1/dropout_7/Cast?
!while/lstm_cell_1/dropout_7/Mul_1Mul#while/lstm_cell_1/dropout_7/Mul:z:0$while/lstm_cell_1/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2#
!while/lstm_cell_1/dropout_7/Mul_1?
while/lstm_cell_1/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/mul?
while/lstm_cell_1/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/mul_1?
while/lstm_cell_1/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/mul_2?
while/lstm_cell_1/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/mul_3t
while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const?
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_1/split/split_dim?
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes
:	P? *
dtype02(
&while/lstm_cell_1/split/ReadVariableOp?
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	P?:	P?:	P?:	P?*
	num_split2
while/lstm_cell_1/split?
while/lstm_cell_1/MatMulMatMulwhile/lstm_cell_1/mul:z:0 while/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul?
while/lstm_cell_1/MatMul_1MatMulwhile/lstm_cell_1/mul_1:z:0 while/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_1?
while/lstm_cell_1/MatMul_2MatMulwhile/lstm_cell_1/mul_2:z:0 while/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_2?
while/lstm_cell_1/MatMul_3MatMulwhile/lstm_cell_1/mul_3:z:0 while/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_3x
while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const_1?
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_1/split_1/split_dim?
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes	
:? *
dtype02*
(while/lstm_cell_1/split_1/ReadVariableOp?
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
while/lstm_cell_1/split_1?
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/BiasAdd?
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/BiasAdd_1?
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/BiasAdd_2?
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/BiasAdd_3?
while/lstm_cell_1/mul_4Mulwhile_placeholder_2%while/lstm_cell_1/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_4?
while/lstm_cell_1/mul_5Mulwhile_placeholder_2%while/lstm_cell_1/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_5?
while/lstm_cell_1/mul_6Mulwhile_placeholder_2%while/lstm_cell_1/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_6?
while/lstm_cell_1/mul_7Mulwhile_placeholder_2%while/lstm_cell_1/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_7?
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02"
 while/lstm_cell_1/ReadVariableOp?
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_1/strided_slice/stack?
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice/stack_1?
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_1/strided_slice/stack_2?
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell_1/strided_slice?
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul_4:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_4?
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add?
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Sigmoid?
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02$
"while/lstm_cell_1/ReadVariableOp_1?
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_1/stack?
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/lstm_cell_1/strided_slice_1/stack_1?
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_1/stack_2?
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_1?
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_5:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_5?
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add_1?
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Sigmoid_1?
while/lstm_cell_1/mul_8Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_8?
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02$
"while/lstm_cell_1/ReadVariableOp_2?
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_2/stack?
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/lstm_cell_1/strided_slice_2/stack_1?
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_2/stack_2?
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_2?
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_6:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_6?
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add_2?
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Tanh?
while/lstm_cell_1/mul_9Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_9?
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_8:z:0while/lstm_cell_1/mul_9:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add_3?
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02$
"while/lstm_cell_1/ReadVariableOp_3?
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_3/stack?
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_1/strided_slice_3/stack_1?
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_3/stack_2?
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_3?
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_7:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_7?
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add_4?
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Sigmoid_2?
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Tanh_1?
while/lstm_cell_1/mul_10Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_10?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_1/mul_10:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_1/add_3:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_lstm_cell_1_layer_call_fn_351338

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_3475802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????P:??????????:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_348894
layer0_input
layer0_348832
layer0_348834
layer0_348836
dense_10_348861
dense_10_348863
output_348888
output_348890
identity?? dense_10/StatefulPartitionedCall?layer0/StatefulPartitionedCall?output/StatefulPartitionedCall?
layer0/StatefulPartitionedCallStatefulPartitionedCalllayer0_inputlayer0_348832layer0_348834layer0_348836*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer0_layer_call_and_return_conditional_losses_3485542 
layer0/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall'layer0/StatefulPartitionedCall:output:0dense_10_348861dense_10_348863*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_3488502"
 dense_10/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0output_348888output_348890*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_3488772 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall^layer0/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????P:::::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2@
layer0/StatefulPartitionedCalllayer0/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:Y U
+
_output_shapes
:?????????P
&
_user_specified_namelayer0_input
??
?
while_body_348673
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
1while_lstm_cell_1_split_readvariableop_resource_07
3while_lstm_cell_1_split_1_readvariableop_resource_0/
+while_lstm_cell_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
/while_lstm_cell_1_split_readvariableop_resource5
1while_lstm_cell_1_split_1_readvariableop_resource-
)while_lstm_cell_1_readvariableop_resource?? while/lstm_cell_1/ReadVariableOp?"while/lstm_cell_1/ReadVariableOp_1?"while/lstm_cell_1/ReadVariableOp_2?"while/lstm_cell_1/ReadVariableOp_3?&while/lstm_cell_1/split/ReadVariableOp?(while/lstm_cell_1/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
!while/lstm_cell_1/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/ones_like/Shape?
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell_1/ones_like/Const?
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/ones_like?
#while/lstm_cell_1/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_1/ones_like_1/Shape?
#while/lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#while/lstm_cell_1/ones_like_1/Const?
while/lstm_cell_1/ones_like_1Fill,while/lstm_cell_1/ones_like_1/Shape:output:0,while/lstm_cell_1/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/ones_like_1?
while/lstm_cell_1/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/mul?
while/lstm_cell_1/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/mul_1?
while/lstm_cell_1/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/mul_2?
while/lstm_cell_1/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/mul_3t
while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const?
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_1/split/split_dim?
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes
:	P? *
dtype02(
&while/lstm_cell_1/split/ReadVariableOp?
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	P?:	P?:	P?:	P?*
	num_split2
while/lstm_cell_1/split?
while/lstm_cell_1/MatMulMatMulwhile/lstm_cell_1/mul:z:0 while/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul?
while/lstm_cell_1/MatMul_1MatMulwhile/lstm_cell_1/mul_1:z:0 while/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_1?
while/lstm_cell_1/MatMul_2MatMulwhile/lstm_cell_1/mul_2:z:0 while/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_2?
while/lstm_cell_1/MatMul_3MatMulwhile/lstm_cell_1/mul_3:z:0 while/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_3x
while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const_1?
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_1/split_1/split_dim?
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes	
:? *
dtype02*
(while/lstm_cell_1/split_1/ReadVariableOp?
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
while/lstm_cell_1/split_1?
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/BiasAdd?
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/BiasAdd_1?
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/BiasAdd_2?
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/BiasAdd_3?
while/lstm_cell_1/mul_4Mulwhile_placeholder_2&while/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_4?
while/lstm_cell_1/mul_5Mulwhile_placeholder_2&while/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_5?
while/lstm_cell_1/mul_6Mulwhile_placeholder_2&while/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_6?
while/lstm_cell_1/mul_7Mulwhile_placeholder_2&while/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_7?
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02"
 while/lstm_cell_1/ReadVariableOp?
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_1/strided_slice/stack?
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice/stack_1?
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_1/strided_slice/stack_2?
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell_1/strided_slice?
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul_4:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_4?
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add?
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Sigmoid?
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02$
"while/lstm_cell_1/ReadVariableOp_1?
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_1/stack?
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/lstm_cell_1/strided_slice_1/stack_1?
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_1/stack_2?
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_1?
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_5:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_5?
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add_1?
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Sigmoid_1?
while/lstm_cell_1/mul_8Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_8?
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02$
"while/lstm_cell_1/ReadVariableOp_2?
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_2/stack?
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/lstm_cell_1/strided_slice_2/stack_1?
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_2/stack_2?
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_2?
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_6:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_6?
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add_2?
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Tanh?
while/lstm_cell_1/mul_9Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_9?
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_8:z:0while/lstm_cell_1/mul_9:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add_3?
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02$
"while/lstm_cell_1/ReadVariableOp_3?
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_3/stack?
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_1/strided_slice_3/stack_1?
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_3/stack_2?
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_3?
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_7:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_7?
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add_4?
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Sigmoid_2?
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Tanh_1?
while/lstm_cell_1/mul_10Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_10?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_1/mul_10:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_1/add_3:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?
B__inference_layer0_layer_call_and_return_conditional_losses_350772

inputs-
)lstm_cell_1_split_readvariableop_resource/
+lstm_cell_1_split_1_readvariableop_resource'
#lstm_cell_1_readvariableop_resource
identity??lstm_cell_1/ReadVariableOp?lstm_cell_1/ReadVariableOp_1?lstm_cell_1/ReadVariableOp_2?lstm_cell_1/ReadVariableOp_3? lstm_cell_1/split/ReadVariableOp?"lstm_cell_1/split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????P2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask2
strided_slice_2?
lstm_cell_1/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like/Shape
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_1/ones_like/Const?
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/ones_like{
lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
lstm_cell_1/dropout/Const?
lstm_cell_1/dropout/MulMullstm_cell_1/ones_like:output:0"lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/dropout/Mul?
lstm_cell_1/dropout/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout/Shape?
0lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2???22
0lstm_cell_1/dropout/random_uniform/RandomUniform?
"lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2$
"lstm_cell_1/dropout/GreaterEqual/y?
 lstm_cell_1/dropout/GreaterEqualGreaterEqual9lstm_cell_1/dropout/random_uniform/RandomUniform:output:0+lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2"
 lstm_cell_1/dropout/GreaterEqual?
lstm_cell_1/dropout/CastCast$lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2
lstm_cell_1/dropout/Cast?
lstm_cell_1/dropout/Mul_1Mullstm_cell_1/dropout/Mul:z:0lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/dropout/Mul_1
lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
lstm_cell_1/dropout_1/Const?
lstm_cell_1/dropout_1/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_1/Mul?
lstm_cell_1/dropout_1/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_1/Shape?
2lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2???24
2lstm_cell_1/dropout_1/random_uniform/RandomUniform?
$lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_1/dropout_1/GreaterEqual/y?
"lstm_cell_1/dropout_1/GreaterEqualGreaterEqual;lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2$
"lstm_cell_1/dropout_1/GreaterEqual?
lstm_cell_1/dropout_1/CastCast&lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_1/Cast?
lstm_cell_1/dropout_1/Mul_1Mullstm_cell_1/dropout_1/Mul:z:0lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_1/Mul_1
lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
lstm_cell_1/dropout_2/Const?
lstm_cell_1/dropout_2/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_2/Mul?
lstm_cell_1/dropout_2/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_2/Shape?
2lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2??r24
2lstm_cell_1/dropout_2/random_uniform/RandomUniform?
$lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_1/dropout_2/GreaterEqual/y?
"lstm_cell_1/dropout_2/GreaterEqualGreaterEqual;lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2$
"lstm_cell_1/dropout_2/GreaterEqual?
lstm_cell_1/dropout_2/CastCast&lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_2/Cast?
lstm_cell_1/dropout_2/Mul_1Mullstm_cell_1/dropout_2/Mul:z:0lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_2/Mul_1
lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
lstm_cell_1/dropout_3/Const?
lstm_cell_1/dropout_3/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_3/Mul?
lstm_cell_1/dropout_3/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_3/Shape?
2lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2롥24
2lstm_cell_1/dropout_3/random_uniform/RandomUniform?
$lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_1/dropout_3/GreaterEqual/y?
"lstm_cell_1/dropout_3/GreaterEqualGreaterEqual;lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2$
"lstm_cell_1/dropout_3/GreaterEqual?
lstm_cell_1/dropout_3/CastCast&lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_3/Cast?
lstm_cell_1/dropout_3/Mul_1Mullstm_cell_1/dropout_3/Mul:z:0lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_3/Mul_1|
lstm_cell_1/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like_1/Shape?
lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_1/ones_like_1/Const?
lstm_cell_1/ones_like_1Fill&lstm_cell_1/ones_like_1/Shape:output:0&lstm_cell_1/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/ones_like_1
lstm_cell_1/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
lstm_cell_1/dropout_4/Const?
lstm_cell_1/dropout_4/MulMul lstm_cell_1/ones_like_1:output:0$lstm_cell_1/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/dropout_4/Mul?
lstm_cell_1/dropout_4/ShapeShape lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_4/Shape?
2lstm_cell_1/dropout_4/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???24
2lstm_cell_1/dropout_4/random_uniform/RandomUniform?
$lstm_cell_1/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_1/dropout_4/GreaterEqual/y?
"lstm_cell_1/dropout_4/GreaterEqualGreaterEqual;lstm_cell_1/dropout_4/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_cell_1/dropout_4/GreaterEqual?
lstm_cell_1/dropout_4/CastCast&lstm_cell_1/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell_1/dropout_4/Cast?
lstm_cell_1/dropout_4/Mul_1Mullstm_cell_1/dropout_4/Mul:z:0lstm_cell_1/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/dropout_4/Mul_1
lstm_cell_1/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
lstm_cell_1/dropout_5/Const?
lstm_cell_1/dropout_5/MulMul lstm_cell_1/ones_like_1:output:0$lstm_cell_1/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/dropout_5/Mul?
lstm_cell_1/dropout_5/ShapeShape lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_5/Shape?
2lstm_cell_1/dropout_5/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2ႄ24
2lstm_cell_1/dropout_5/random_uniform/RandomUniform?
$lstm_cell_1/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_1/dropout_5/GreaterEqual/y?
"lstm_cell_1/dropout_5/GreaterEqualGreaterEqual;lstm_cell_1/dropout_5/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_cell_1/dropout_5/GreaterEqual?
lstm_cell_1/dropout_5/CastCast&lstm_cell_1/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell_1/dropout_5/Cast?
lstm_cell_1/dropout_5/Mul_1Mullstm_cell_1/dropout_5/Mul:z:0lstm_cell_1/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/dropout_5/Mul_1
lstm_cell_1/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
lstm_cell_1/dropout_6/Const?
lstm_cell_1/dropout_6/MulMul lstm_cell_1/ones_like_1:output:0$lstm_cell_1/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/dropout_6/Mul?
lstm_cell_1/dropout_6/ShapeShape lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_6/Shape?
2lstm_cell_1/dropout_6/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2ϛX24
2lstm_cell_1/dropout_6/random_uniform/RandomUniform?
$lstm_cell_1/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_1/dropout_6/GreaterEqual/y?
"lstm_cell_1/dropout_6/GreaterEqualGreaterEqual;lstm_cell_1/dropout_6/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_cell_1/dropout_6/GreaterEqual?
lstm_cell_1/dropout_6/CastCast&lstm_cell_1/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell_1/dropout_6/Cast?
lstm_cell_1/dropout_6/Mul_1Mullstm_cell_1/dropout_6/Mul:z:0lstm_cell_1/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/dropout_6/Mul_1
lstm_cell_1/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
lstm_cell_1/dropout_7/Const?
lstm_cell_1/dropout_7/MulMul lstm_cell_1/ones_like_1:output:0$lstm_cell_1/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/dropout_7/Mul?
lstm_cell_1/dropout_7/ShapeShape lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_7/Shape?
2lstm_cell_1/dropout_7/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???24
2lstm_cell_1/dropout_7/random_uniform/RandomUniform?
$lstm_cell_1/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_1/dropout_7/GreaterEqual/y?
"lstm_cell_1/dropout_7/GreaterEqualGreaterEqual;lstm_cell_1/dropout_7/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_cell_1/dropout_7/GreaterEqual?
lstm_cell_1/dropout_7/CastCast&lstm_cell_1/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell_1/dropout_7/Cast?
lstm_cell_1/dropout_7/Mul_1Mullstm_cell_1/dropout_7/Mul:z:0lstm_cell_1/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/dropout_7/Mul_1?
lstm_cell_1/mulMulstrided_slice_2:output:0lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/mul?
lstm_cell_1/mul_1Mulstrided_slice_2:output:0lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/mul_1?
lstm_cell_1/mul_2Mulstrided_slice_2:output:0lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/mul_2?
lstm_cell_1/mul_3Mulstrided_slice_2:output:0lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/mul_3h
lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dim?
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource*
_output_shapes
:	P? *
dtype02"
 lstm_cell_1/split/ReadVariableOp?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	P?:	P?:	P?:	P?*
	num_split2
lstm_cell_1/split?
lstm_cell_1/MatMulMatMullstm_cell_1/mul:z:0lstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul?
lstm_cell_1/MatMul_1MatMullstm_cell_1/mul_1:z:0lstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_1?
lstm_cell_1/MatMul_2MatMullstm_cell_1/mul_2:z:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_2?
lstm_cell_1/MatMul_3MatMullstm_cell_1/mul_3:z:0lstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_3l
lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const_1?
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_1/split_1/split_dim?
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes	
:? *
dtype02$
"lstm_cell_1/split_1/ReadVariableOp?
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell_1/split_1?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/BiasAdd?
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_1/BiasAdd_1?
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_1/BiasAdd_2?
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_1/BiasAdd_3?
lstm_cell_1/mul_4Mulzeros:output:0lstm_cell_1/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_4?
lstm_cell_1/mul_5Mulzeros:output:0lstm_cell_1/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_5?
lstm_cell_1/mul_6Mulzeros:output:0lstm_cell_1/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_6?
lstm_cell_1/mul_7Mulzeros:output:0lstm_cell_1/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_7?
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
lstm_cell_1/ReadVariableOp?
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_1/strided_slice/stack?
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice/stack_1?
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_1/strided_slice/stack_2?
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_1/strided_slice?
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul_4:z:0"lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_4?
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add}
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Sigmoid?
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
lstm_cell_1/ReadVariableOp_1?
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_1/stack?
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#lstm_cell_1/strided_slice_1/stack_1?
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_1/stack_2?
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_1?
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_5:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_5?
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add_1?
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Sigmoid_1?
lstm_cell_1/mul_8Mullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_8?
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
lstm_cell_1/ReadVariableOp_2?
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_2/stack?
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#lstm_cell_1/strided_slice_2/stack_1?
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_2/stack_2?
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_2?
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_6:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_6?
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add_2v
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Tanh?
lstm_cell_1/mul_9Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_9?
lstm_cell_1/add_3AddV2lstm_cell_1/mul_8:z:0lstm_cell_1/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add_3?
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
lstm_cell_1/ReadVariableOp_3?
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_3/stack?
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_1/strided_slice_3/stack_1?
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_3/stack_2?
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_3?
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_7:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_7?
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add_4?
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Sigmoid_2z
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Tanh_1?
lstm_cell_1/mul_10Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_350572*
condR
while_cond_350571*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????P:::28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?

?
layer0_while_cond_349207*
&layer0_while_layer0_while_loop_counter0
,layer0_while_layer0_while_maximum_iterations
layer0_while_placeholder
layer0_while_placeholder_1
layer0_while_placeholder_2
layer0_while_placeholder_3,
(layer0_while_less_layer0_strided_slice_1B
>layer0_while_layer0_while_cond_349207___redundant_placeholder0B
>layer0_while_layer0_while_cond_349207___redundant_placeholder1B
>layer0_while_layer0_while_cond_349207___redundant_placeholder2B
>layer0_while_layer0_while_cond_349207___redundant_placeholder3
layer0_while_identity
?
layer0/while/LessLesslayer0_while_placeholder(layer0_while_less_layer0_strided_slice_1*
T0*
_output_shapes
: 2
layer0/while/Lessr
layer0/while/IdentityIdentitylayer0/while/Less:z:0*
T0
*
_output_shapes
: 2
layer0/while/Identity"7
layer0_while_identitylayer0/while/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
,__inference_lstm_cell_1_layer_call_fn_351355

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_3476642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????P:??????????:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
??
?
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_351237

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpX
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????P2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:?????????P2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2???2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????P2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_1/Const?
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????P2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2???2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_1/GreaterEqual/y?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2
dropout_1/GreaterEqual?
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2
dropout_1/Cast?
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????P2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_2/Const?
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????P2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2???2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_2/GreaterEqual/y?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2
dropout_2/GreaterEqual?
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2
dropout_2/Cast?
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????P2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_3/Const?
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????P2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shape?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2???2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_3/GreaterEqual/y?
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2
dropout_3/GreaterEqual?
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2
dropout_3/Cast?
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????P2
dropout_3/Mul_1^
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
ones_like_1g
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_4/Const?
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_4/Mulf
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_4/Shape?
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??02(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_4/GreaterEqual/y?
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_4/GreaterEqual?
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_4/Cast?
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_4/Mul_1g
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_5/Const?
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_5/Mulf
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_5/Shape?
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ڒ2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_5/GreaterEqual/y?
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_5/GreaterEqual?
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_5/Cast?
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_5/Mul_1g
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_6/Const?
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_6/Mulf
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_6/Shape?
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2(
&dropout_6/random_uniform/RandomUniformy
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_6/GreaterEqual/y?
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_6/GreaterEqual?
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_6/Cast?
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_6/Mul_1g
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_7/Const?
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_7/Mulf
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_7/Shape?
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??T2(
&dropout_7/random_uniform/RandomUniformy
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_7/GreaterEqual/y?
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_7/GreaterEqual?
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_7/Cast?
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_7/Mul_1^
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
muld
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
mul_1d
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
mul_2d
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	P? *
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	P?:	P?:	P?:	P?*
	num_split2
splitf
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:??????????2
MatMull
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:??????????2

MatMul_1l
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:??????????2

MatMul_2l
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:??????????2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:? *
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????2
	BiasAdd_3g
mul_4Mulstates_0dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_4g
mul_5Mulstates_0dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_5g
mul_6Mulstates_0dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_6g
mul_7Mulstates_0dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_7z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
?? *
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slicet
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
?? *
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_1v
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1a
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mul_8~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
?? *
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_2v
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????2
Tanh_
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????2
mul_9`
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:??????????2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
?? *
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_3v
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:??????????2
Tanh_1e
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
mul_10?
IdentityIdentity
mul_10:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity
mul_10:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????P:??????????:??????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
??
?
B__inference_layer0_layer_call_and_return_conditional_losses_348554

inputs-
)lstm_cell_1_split_readvariableop_resource/
+lstm_cell_1_split_1_readvariableop_resource'
#lstm_cell_1_readvariableop_resource
identity??lstm_cell_1/ReadVariableOp?lstm_cell_1/ReadVariableOp_1?lstm_cell_1/ReadVariableOp_2?lstm_cell_1/ReadVariableOp_3? lstm_cell_1/split/ReadVariableOp?"lstm_cell_1/split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????P2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask2
strided_slice_2?
lstm_cell_1/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like/Shape
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_1/ones_like/Const?
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/ones_like{
lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
lstm_cell_1/dropout/Const?
lstm_cell_1/dropout/MulMullstm_cell_1/ones_like:output:0"lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/dropout/Mul?
lstm_cell_1/dropout/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout/Shape?
0lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2???22
0lstm_cell_1/dropout/random_uniform/RandomUniform?
"lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2$
"lstm_cell_1/dropout/GreaterEqual/y?
 lstm_cell_1/dropout/GreaterEqualGreaterEqual9lstm_cell_1/dropout/random_uniform/RandomUniform:output:0+lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2"
 lstm_cell_1/dropout/GreaterEqual?
lstm_cell_1/dropout/CastCast$lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2
lstm_cell_1/dropout/Cast?
lstm_cell_1/dropout/Mul_1Mullstm_cell_1/dropout/Mul:z:0lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/dropout/Mul_1
lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
lstm_cell_1/dropout_1/Const?
lstm_cell_1/dropout_1/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_1/Mul?
lstm_cell_1/dropout_1/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_1/Shape?
2lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2???24
2lstm_cell_1/dropout_1/random_uniform/RandomUniform?
$lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_1/dropout_1/GreaterEqual/y?
"lstm_cell_1/dropout_1/GreaterEqualGreaterEqual;lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2$
"lstm_cell_1/dropout_1/GreaterEqual?
lstm_cell_1/dropout_1/CastCast&lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_1/Cast?
lstm_cell_1/dropout_1/Mul_1Mullstm_cell_1/dropout_1/Mul:z:0lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_1/Mul_1
lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
lstm_cell_1/dropout_2/Const?
lstm_cell_1/dropout_2/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_2/Mul?
lstm_cell_1/dropout_2/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_2/Shape?
2lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2???24
2lstm_cell_1/dropout_2/random_uniform/RandomUniform?
$lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_1/dropout_2/GreaterEqual/y?
"lstm_cell_1/dropout_2/GreaterEqualGreaterEqual;lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2$
"lstm_cell_1/dropout_2/GreaterEqual?
lstm_cell_1/dropout_2/CastCast&lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_2/Cast?
lstm_cell_1/dropout_2/Mul_1Mullstm_cell_1/dropout_2/Mul:z:0lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_2/Mul_1
lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
lstm_cell_1/dropout_3/Const?
lstm_cell_1/dropout_3/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_3/Mul?
lstm_cell_1/dropout_3/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_3/Shape?
2lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2???24
2lstm_cell_1/dropout_3/random_uniform/RandomUniform?
$lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_1/dropout_3/GreaterEqual/y?
"lstm_cell_1/dropout_3/GreaterEqualGreaterEqual;lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2$
"lstm_cell_1/dropout_3/GreaterEqual?
lstm_cell_1/dropout_3/CastCast&lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_3/Cast?
lstm_cell_1/dropout_3/Mul_1Mullstm_cell_1/dropout_3/Mul:z:0lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_3/Mul_1|
lstm_cell_1/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like_1/Shape?
lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_1/ones_like_1/Const?
lstm_cell_1/ones_like_1Fill&lstm_cell_1/ones_like_1/Shape:output:0&lstm_cell_1/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/ones_like_1
lstm_cell_1/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
lstm_cell_1/dropout_4/Const?
lstm_cell_1/dropout_4/MulMul lstm_cell_1/ones_like_1:output:0$lstm_cell_1/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/dropout_4/Mul?
lstm_cell_1/dropout_4/ShapeShape lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_4/Shape?
2lstm_cell_1/dropout_4/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???24
2lstm_cell_1/dropout_4/random_uniform/RandomUniform?
$lstm_cell_1/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_1/dropout_4/GreaterEqual/y?
"lstm_cell_1/dropout_4/GreaterEqualGreaterEqual;lstm_cell_1/dropout_4/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_cell_1/dropout_4/GreaterEqual?
lstm_cell_1/dropout_4/CastCast&lstm_cell_1/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell_1/dropout_4/Cast?
lstm_cell_1/dropout_4/Mul_1Mullstm_cell_1/dropout_4/Mul:z:0lstm_cell_1/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/dropout_4/Mul_1
lstm_cell_1/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
lstm_cell_1/dropout_5/Const?
lstm_cell_1/dropout_5/MulMul lstm_cell_1/ones_like_1:output:0$lstm_cell_1/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/dropout_5/Mul?
lstm_cell_1/dropout_5/ShapeShape lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_5/Shape?
2lstm_cell_1/dropout_5/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ܕ24
2lstm_cell_1/dropout_5/random_uniform/RandomUniform?
$lstm_cell_1/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_1/dropout_5/GreaterEqual/y?
"lstm_cell_1/dropout_5/GreaterEqualGreaterEqual;lstm_cell_1/dropout_5/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_cell_1/dropout_5/GreaterEqual?
lstm_cell_1/dropout_5/CastCast&lstm_cell_1/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell_1/dropout_5/Cast?
lstm_cell_1/dropout_5/Mul_1Mullstm_cell_1/dropout_5/Mul:z:0lstm_cell_1/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/dropout_5/Mul_1
lstm_cell_1/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
lstm_cell_1/dropout_6/Const?
lstm_cell_1/dropout_6/MulMul lstm_cell_1/ones_like_1:output:0$lstm_cell_1/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/dropout_6/Mul?
lstm_cell_1/dropout_6/ShapeShape lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_6/Shape?
2lstm_cell_1/dropout_6/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??=24
2lstm_cell_1/dropout_6/random_uniform/RandomUniform?
$lstm_cell_1/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_1/dropout_6/GreaterEqual/y?
"lstm_cell_1/dropout_6/GreaterEqualGreaterEqual;lstm_cell_1/dropout_6/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_cell_1/dropout_6/GreaterEqual?
lstm_cell_1/dropout_6/CastCast&lstm_cell_1/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell_1/dropout_6/Cast?
lstm_cell_1/dropout_6/Mul_1Mullstm_cell_1/dropout_6/Mul:z:0lstm_cell_1/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/dropout_6/Mul_1
lstm_cell_1/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
lstm_cell_1/dropout_7/Const?
lstm_cell_1/dropout_7/MulMul lstm_cell_1/ones_like_1:output:0$lstm_cell_1/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/dropout_7/Mul?
lstm_cell_1/dropout_7/ShapeShape lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_7/Shape?
2lstm_cell_1/dropout_7/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???24
2lstm_cell_1/dropout_7/random_uniform/RandomUniform?
$lstm_cell_1/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_1/dropout_7/GreaterEqual/y?
"lstm_cell_1/dropout_7/GreaterEqualGreaterEqual;lstm_cell_1/dropout_7/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_cell_1/dropout_7/GreaterEqual?
lstm_cell_1/dropout_7/CastCast&lstm_cell_1/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell_1/dropout_7/Cast?
lstm_cell_1/dropout_7/Mul_1Mullstm_cell_1/dropout_7/Mul:z:0lstm_cell_1/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/dropout_7/Mul_1?
lstm_cell_1/mulMulstrided_slice_2:output:0lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/mul?
lstm_cell_1/mul_1Mulstrided_slice_2:output:0lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/mul_1?
lstm_cell_1/mul_2Mulstrided_slice_2:output:0lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/mul_2?
lstm_cell_1/mul_3Mulstrided_slice_2:output:0lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/mul_3h
lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dim?
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource*
_output_shapes
:	P? *
dtype02"
 lstm_cell_1/split/ReadVariableOp?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	P?:	P?:	P?:	P?*
	num_split2
lstm_cell_1/split?
lstm_cell_1/MatMulMatMullstm_cell_1/mul:z:0lstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul?
lstm_cell_1/MatMul_1MatMullstm_cell_1/mul_1:z:0lstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_1?
lstm_cell_1/MatMul_2MatMullstm_cell_1/mul_2:z:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_2?
lstm_cell_1/MatMul_3MatMullstm_cell_1/mul_3:z:0lstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_3l
lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const_1?
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_1/split_1/split_dim?
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes	
:? *
dtype02$
"lstm_cell_1/split_1/ReadVariableOp?
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell_1/split_1?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/BiasAdd?
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_1/BiasAdd_1?
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_1/BiasAdd_2?
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_1/BiasAdd_3?
lstm_cell_1/mul_4Mulzeros:output:0lstm_cell_1/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_4?
lstm_cell_1/mul_5Mulzeros:output:0lstm_cell_1/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_5?
lstm_cell_1/mul_6Mulzeros:output:0lstm_cell_1/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_6?
lstm_cell_1/mul_7Mulzeros:output:0lstm_cell_1/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_7?
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
lstm_cell_1/ReadVariableOp?
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_1/strided_slice/stack?
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice/stack_1?
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_1/strided_slice/stack_2?
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_1/strided_slice?
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul_4:z:0"lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_4?
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add}
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Sigmoid?
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
lstm_cell_1/ReadVariableOp_1?
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_1/stack?
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#lstm_cell_1/strided_slice_1/stack_1?
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_1/stack_2?
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_1?
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_5:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_5?
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add_1?
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Sigmoid_1?
lstm_cell_1/mul_8Mullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_8?
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
lstm_cell_1/ReadVariableOp_2?
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_2/stack?
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#lstm_cell_1/strided_slice_2/stack_1?
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_2/stack_2?
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_2?
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_6:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_6?
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add_2v
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Tanh?
lstm_cell_1/mul_9Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_9?
lstm_cell_1/add_3AddV2lstm_cell_1/mul_8:z:0lstm_cell_1/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add_3?
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
lstm_cell_1/ReadVariableOp_3?
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_3/stack?
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_1/strided_slice_3/stack_1?
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_3/stack_2?
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_3?
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_7:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_7?
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add_4?
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Sigmoid_2z
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Tanh_1?
lstm_cell_1/mul_10Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_348354*
condR
while_cond_348353*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????P:::28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
??
?
while_body_350231
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
1while_lstm_cell_1_split_readvariableop_resource_07
3while_lstm_cell_1_split_1_readvariableop_resource_0/
+while_lstm_cell_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
/while_lstm_cell_1_split_readvariableop_resource5
1while_lstm_cell_1_split_1_readvariableop_resource-
)while_lstm_cell_1_readvariableop_resource?? while/lstm_cell_1/ReadVariableOp?"while/lstm_cell_1/ReadVariableOp_1?"while/lstm_cell_1/ReadVariableOp_2?"while/lstm_cell_1/ReadVariableOp_3?&while/lstm_cell_1/split/ReadVariableOp?(while/lstm_cell_1/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
!while/lstm_cell_1/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/ones_like/Shape?
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell_1/ones_like/Const?
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/ones_like?
#while/lstm_cell_1/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_1/ones_like_1/Shape?
#while/lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#while/lstm_cell_1/ones_like_1/Const?
while/lstm_cell_1/ones_like_1Fill,while/lstm_cell_1/ones_like_1/Shape:output:0,while/lstm_cell_1/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/ones_like_1?
while/lstm_cell_1/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/mul?
while/lstm_cell_1/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/mul_1?
while/lstm_cell_1/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/mul_2?
while/lstm_cell_1/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/mul_3t
while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const?
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_1/split/split_dim?
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes
:	P? *
dtype02(
&while/lstm_cell_1/split/ReadVariableOp?
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	P?:	P?:	P?:	P?*
	num_split2
while/lstm_cell_1/split?
while/lstm_cell_1/MatMulMatMulwhile/lstm_cell_1/mul:z:0 while/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul?
while/lstm_cell_1/MatMul_1MatMulwhile/lstm_cell_1/mul_1:z:0 while/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_1?
while/lstm_cell_1/MatMul_2MatMulwhile/lstm_cell_1/mul_2:z:0 while/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_2?
while/lstm_cell_1/MatMul_3MatMulwhile/lstm_cell_1/mul_3:z:0 while/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_3x
while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const_1?
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_1/split_1/split_dim?
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes	
:? *
dtype02*
(while/lstm_cell_1/split_1/ReadVariableOp?
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
while/lstm_cell_1/split_1?
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/BiasAdd?
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/BiasAdd_1?
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/BiasAdd_2?
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/BiasAdd_3?
while/lstm_cell_1/mul_4Mulwhile_placeholder_2&while/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_4?
while/lstm_cell_1/mul_5Mulwhile_placeholder_2&while/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_5?
while/lstm_cell_1/mul_6Mulwhile_placeholder_2&while/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_6?
while/lstm_cell_1/mul_7Mulwhile_placeholder_2&while/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_7?
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02"
 while/lstm_cell_1/ReadVariableOp?
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_1/strided_slice/stack?
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice/stack_1?
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_1/strided_slice/stack_2?
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell_1/strided_slice?
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul_4:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_4?
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add?
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Sigmoid?
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02$
"while/lstm_cell_1/ReadVariableOp_1?
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_1/stack?
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/lstm_cell_1/strided_slice_1/stack_1?
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_1/stack_2?
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_1?
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_5:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_5?
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add_1?
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Sigmoid_1?
while/lstm_cell_1/mul_8Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_8?
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02$
"while/lstm_cell_1/ReadVariableOp_2?
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_2/stack?
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/lstm_cell_1/strided_slice_2/stack_1?
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_2/stack_2?
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_2?
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_6:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_6?
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add_2?
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Tanh?
while/lstm_cell_1/mul_9Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_9?
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_8:z:0while/lstm_cell_1/mul_9:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add_3?
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02$
"while/lstm_cell_1/ReadVariableOp_3?
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_3/stack?
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_1/strided_slice_3/stack_1?
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_3/stack_2?
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_3?
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_7:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_7?
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add_4?
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Sigmoid_2?
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Tanh_1?
while/lstm_cell_1/mul_10Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_10?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_1/mul_10:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_1/add_3:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
.__inference_sequential_12_layer_call_fn_349729

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_3489792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????P:::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
|
'__inference_output_layer_call_fn_351089

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_3488772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
layer0_while_cond_349540*
&layer0_while_layer0_while_loop_counter0
,layer0_while_layer0_while_maximum_iterations
layer0_while_placeholder
layer0_while_placeholder_1
layer0_while_placeholder_2
layer0_while_placeholder_3,
(layer0_while_less_layer0_strided_slice_1B
>layer0_while_layer0_while_cond_349540___redundant_placeholder0B
>layer0_while_layer0_while_cond_349540___redundant_placeholder1B
>layer0_while_layer0_while_cond_349540___redundant_placeholder2B
>layer0_while_layer0_while_cond_349540___redundant_placeholder3
layer0_while_identity
?
layer0/while/LessLesslayer0_while_placeholder(layer0_while_less_layer0_strided_slice_1*
T0*
_output_shapes
: 2
layer0/while/Lessr
layer0/while/IdentityIdentitylayer0/while/Less:z:0*
T0
*
_output_shapes
: 2
layer0/while/Identity"7
layer0_while_identitylayer0/while/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
&sequential_12_layer0_while_body_347242F
Bsequential_12_layer0_while_sequential_12_layer0_while_loop_counterL
Hsequential_12_layer0_while_sequential_12_layer0_while_maximum_iterations*
&sequential_12_layer0_while_placeholder,
(sequential_12_layer0_while_placeholder_1,
(sequential_12_layer0_while_placeholder_2,
(sequential_12_layer0_while_placeholder_3E
Asequential_12_layer0_while_sequential_12_layer0_strided_slice_1_0?
}sequential_12_layer0_while_tensorarrayv2read_tensorlistgetitem_sequential_12_layer0_tensorarrayunstack_tensorlistfromtensor_0J
Fsequential_12_layer0_while_lstm_cell_1_split_readvariableop_resource_0L
Hsequential_12_layer0_while_lstm_cell_1_split_1_readvariableop_resource_0D
@sequential_12_layer0_while_lstm_cell_1_readvariableop_resource_0'
#sequential_12_layer0_while_identity)
%sequential_12_layer0_while_identity_1)
%sequential_12_layer0_while_identity_2)
%sequential_12_layer0_while_identity_3)
%sequential_12_layer0_while_identity_4)
%sequential_12_layer0_while_identity_5C
?sequential_12_layer0_while_sequential_12_layer0_strided_slice_1
{sequential_12_layer0_while_tensorarrayv2read_tensorlistgetitem_sequential_12_layer0_tensorarrayunstack_tensorlistfromtensorH
Dsequential_12_layer0_while_lstm_cell_1_split_readvariableop_resourceJ
Fsequential_12_layer0_while_lstm_cell_1_split_1_readvariableop_resourceB
>sequential_12_layer0_while_lstm_cell_1_readvariableop_resource??5sequential_12/layer0/while/lstm_cell_1/ReadVariableOp?7sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_1?7sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_2?7sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_3?;sequential_12/layer0/while/lstm_cell_1/split/ReadVariableOp?=sequential_12/layer0/while/lstm_cell_1/split_1/ReadVariableOp?
Lsequential_12/layer0/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   2N
Lsequential_12/layer0/while/TensorArrayV2Read/TensorListGetItem/element_shape?
>sequential_12/layer0/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_12_layer0_while_tensorarrayv2read_tensorlistgetitem_sequential_12_layer0_tensorarrayunstack_tensorlistfromtensor_0&sequential_12_layer0_while_placeholderUsequential_12/layer0/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype02@
>sequential_12/layer0/while/TensorArrayV2Read/TensorListGetItem?
6sequential_12/layer0/while/lstm_cell_1/ones_like/ShapeShapeEsequential_12/layer0/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:28
6sequential_12/layer0/while/lstm_cell_1/ones_like/Shape?
6sequential_12/layer0/while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??28
6sequential_12/layer0/while/lstm_cell_1/ones_like/Const?
0sequential_12/layer0/while/lstm_cell_1/ones_likeFill?sequential_12/layer0/while/lstm_cell_1/ones_like/Shape:output:0?sequential_12/layer0/while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????P22
0sequential_12/layer0/while/lstm_cell_1/ones_like?
8sequential_12/layer0/while/lstm_cell_1/ones_like_1/ShapeShape(sequential_12_layer0_while_placeholder_2*
T0*
_output_shapes
:2:
8sequential_12/layer0/while/lstm_cell_1/ones_like_1/Shape?
8sequential_12/layer0/while/lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2:
8sequential_12/layer0/while/lstm_cell_1/ones_like_1/Const?
2sequential_12/layer0/while/lstm_cell_1/ones_like_1FillAsequential_12/layer0/while/lstm_cell_1/ones_like_1/Shape:output:0Asequential_12/layer0/while/lstm_cell_1/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????24
2sequential_12/layer0/while/lstm_cell_1/ones_like_1?
*sequential_12/layer0/while/lstm_cell_1/mulMulEsequential_12/layer0/while/TensorArrayV2Read/TensorListGetItem:item:09sequential_12/layer0/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2,
*sequential_12/layer0/while/lstm_cell_1/mul?
,sequential_12/layer0/while/lstm_cell_1/mul_1MulEsequential_12/layer0/while/TensorArrayV2Read/TensorListGetItem:item:09sequential_12/layer0/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2.
,sequential_12/layer0/while/lstm_cell_1/mul_1?
,sequential_12/layer0/while/lstm_cell_1/mul_2MulEsequential_12/layer0/while/TensorArrayV2Read/TensorListGetItem:item:09sequential_12/layer0/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2.
,sequential_12/layer0/while/lstm_cell_1/mul_2?
,sequential_12/layer0/while/lstm_cell_1/mul_3MulEsequential_12/layer0/while/TensorArrayV2Read/TensorListGetItem:item:09sequential_12/layer0/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2.
,sequential_12/layer0/while/lstm_cell_1/mul_3?
,sequential_12/layer0/while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2.
,sequential_12/layer0/while/lstm_cell_1/Const?
6sequential_12/layer0/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential_12/layer0/while/lstm_cell_1/split/split_dim?
;sequential_12/layer0/while/lstm_cell_1/split/ReadVariableOpReadVariableOpFsequential_12_layer0_while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes
:	P? *
dtype02=
;sequential_12/layer0/while/lstm_cell_1/split/ReadVariableOp?
,sequential_12/layer0/while/lstm_cell_1/splitSplit?sequential_12/layer0/while/lstm_cell_1/split/split_dim:output:0Csequential_12/layer0/while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	P?:	P?:	P?:	P?*
	num_split2.
,sequential_12/layer0/while/lstm_cell_1/split?
-sequential_12/layer0/while/lstm_cell_1/MatMulMatMul.sequential_12/layer0/while/lstm_cell_1/mul:z:05sequential_12/layer0/while/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????2/
-sequential_12/layer0/while/lstm_cell_1/MatMul?
/sequential_12/layer0/while/lstm_cell_1/MatMul_1MatMul0sequential_12/layer0/while/lstm_cell_1/mul_1:z:05sequential_12/layer0/while/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????21
/sequential_12/layer0/while/lstm_cell_1/MatMul_1?
/sequential_12/layer0/while/lstm_cell_1/MatMul_2MatMul0sequential_12/layer0/while/lstm_cell_1/mul_2:z:05sequential_12/layer0/while/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????21
/sequential_12/layer0/while/lstm_cell_1/MatMul_2?
/sequential_12/layer0/while/lstm_cell_1/MatMul_3MatMul0sequential_12/layer0/while/lstm_cell_1/mul_3:z:05sequential_12/layer0/while/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????21
/sequential_12/layer0/while/lstm_cell_1/MatMul_3?
.sequential_12/layer0/while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :20
.sequential_12/layer0/while/lstm_cell_1/Const_1?
8sequential_12/layer0/while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2:
8sequential_12/layer0/while/lstm_cell_1/split_1/split_dim?
=sequential_12/layer0/while/lstm_cell_1/split_1/ReadVariableOpReadVariableOpHsequential_12_layer0_while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes	
:? *
dtype02?
=sequential_12/layer0/while/lstm_cell_1/split_1/ReadVariableOp?
.sequential_12/layer0/while/lstm_cell_1/split_1SplitAsequential_12/layer0/while/lstm_cell_1/split_1/split_dim:output:0Esequential_12/layer0/while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split20
.sequential_12/layer0/while/lstm_cell_1/split_1?
.sequential_12/layer0/while/lstm_cell_1/BiasAddBiasAdd7sequential_12/layer0/while/lstm_cell_1/MatMul:product:07sequential_12/layer0/while/lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:??????????20
.sequential_12/layer0/while/lstm_cell_1/BiasAdd?
0sequential_12/layer0/while/lstm_cell_1/BiasAdd_1BiasAdd9sequential_12/layer0/while/lstm_cell_1/MatMul_1:product:07sequential_12/layer0/while/lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:??????????22
0sequential_12/layer0/while/lstm_cell_1/BiasAdd_1?
0sequential_12/layer0/while/lstm_cell_1/BiasAdd_2BiasAdd9sequential_12/layer0/while/lstm_cell_1/MatMul_2:product:07sequential_12/layer0/while/lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:??????????22
0sequential_12/layer0/while/lstm_cell_1/BiasAdd_2?
0sequential_12/layer0/while/lstm_cell_1/BiasAdd_3BiasAdd9sequential_12/layer0/while/lstm_cell_1/MatMul_3:product:07sequential_12/layer0/while/lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:??????????22
0sequential_12/layer0/while/lstm_cell_1/BiasAdd_3?
,sequential_12/layer0/while/lstm_cell_1/mul_4Mul(sequential_12_layer0_while_placeholder_2;sequential_12/layer0/while/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2.
,sequential_12/layer0/while/lstm_cell_1/mul_4?
,sequential_12/layer0/while/lstm_cell_1/mul_5Mul(sequential_12_layer0_while_placeholder_2;sequential_12/layer0/while/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2.
,sequential_12/layer0/while/lstm_cell_1/mul_5?
,sequential_12/layer0/while/lstm_cell_1/mul_6Mul(sequential_12_layer0_while_placeholder_2;sequential_12/layer0/while/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2.
,sequential_12/layer0/while/lstm_cell_1/mul_6?
,sequential_12/layer0/while/lstm_cell_1/mul_7Mul(sequential_12_layer0_while_placeholder_2;sequential_12/layer0/while/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2.
,sequential_12/layer0/while/lstm_cell_1/mul_7?
5sequential_12/layer0/while/lstm_cell_1/ReadVariableOpReadVariableOp@sequential_12_layer0_while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype027
5sequential_12/layer0/while/lstm_cell_1/ReadVariableOp?
:sequential_12/layer0/while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2<
:sequential_12/layer0/while/lstm_cell_1/strided_slice/stack?
<sequential_12/layer0/while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2>
<sequential_12/layer0/while/lstm_cell_1/strided_slice/stack_1?
<sequential_12/layer0/while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2>
<sequential_12/layer0/while/lstm_cell_1/strided_slice/stack_2?
4sequential_12/layer0/while/lstm_cell_1/strided_sliceStridedSlice=sequential_12/layer0/while/lstm_cell_1/ReadVariableOp:value:0Csequential_12/layer0/while/lstm_cell_1/strided_slice/stack:output:0Esequential_12/layer0/while/lstm_cell_1/strided_slice/stack_1:output:0Esequential_12/layer0/while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask26
4sequential_12/layer0/while/lstm_cell_1/strided_slice?
/sequential_12/layer0/while/lstm_cell_1/MatMul_4MatMul0sequential_12/layer0/while/lstm_cell_1/mul_4:z:0=sequential_12/layer0/while/lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????21
/sequential_12/layer0/while/lstm_cell_1/MatMul_4?
*sequential_12/layer0/while/lstm_cell_1/addAddV27sequential_12/layer0/while/lstm_cell_1/BiasAdd:output:09sequential_12/layer0/while/lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2,
*sequential_12/layer0/while/lstm_cell_1/add?
.sequential_12/layer0/while/lstm_cell_1/SigmoidSigmoid.sequential_12/layer0/while/lstm_cell_1/add:z:0*
T0*(
_output_shapes
:??????????20
.sequential_12/layer0/while/lstm_cell_1/Sigmoid?
7sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_1ReadVariableOp@sequential_12_layer0_while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype029
7sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_1?
<sequential_12/layer0/while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2>
<sequential_12/layer0/while/lstm_cell_1/strided_slice_1/stack?
>sequential_12/layer0/while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2@
>sequential_12/layer0/while/lstm_cell_1/strided_slice_1/stack_1?
>sequential_12/layer0/while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2@
>sequential_12/layer0/while/lstm_cell_1/strided_slice_1/stack_2?
6sequential_12/layer0/while/lstm_cell_1/strided_slice_1StridedSlice?sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_1:value:0Esequential_12/layer0/while/lstm_cell_1/strided_slice_1/stack:output:0Gsequential_12/layer0/while/lstm_cell_1/strided_slice_1/stack_1:output:0Gsequential_12/layer0/while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask28
6sequential_12/layer0/while/lstm_cell_1/strided_slice_1?
/sequential_12/layer0/while/lstm_cell_1/MatMul_5MatMul0sequential_12/layer0/while/lstm_cell_1/mul_5:z:0?sequential_12/layer0/while/lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????21
/sequential_12/layer0/while/lstm_cell_1/MatMul_5?
,sequential_12/layer0/while/lstm_cell_1/add_1AddV29sequential_12/layer0/while/lstm_cell_1/BiasAdd_1:output:09sequential_12/layer0/while/lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2.
,sequential_12/layer0/while/lstm_cell_1/add_1?
0sequential_12/layer0/while/lstm_cell_1/Sigmoid_1Sigmoid0sequential_12/layer0/while/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????22
0sequential_12/layer0/while/lstm_cell_1/Sigmoid_1?
,sequential_12/layer0/while/lstm_cell_1/mul_8Mul4sequential_12/layer0/while/lstm_cell_1/Sigmoid_1:y:0(sequential_12_layer0_while_placeholder_3*
T0*(
_output_shapes
:??????????2.
,sequential_12/layer0/while/lstm_cell_1/mul_8?
7sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_2ReadVariableOp@sequential_12_layer0_while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype029
7sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_2?
<sequential_12/layer0/while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2>
<sequential_12/layer0/while/lstm_cell_1/strided_slice_2/stack?
>sequential_12/layer0/while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2@
>sequential_12/layer0/while/lstm_cell_1/strided_slice_2/stack_1?
>sequential_12/layer0/while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2@
>sequential_12/layer0/while/lstm_cell_1/strided_slice_2/stack_2?
6sequential_12/layer0/while/lstm_cell_1/strided_slice_2StridedSlice?sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_2:value:0Esequential_12/layer0/while/lstm_cell_1/strided_slice_2/stack:output:0Gsequential_12/layer0/while/lstm_cell_1/strided_slice_2/stack_1:output:0Gsequential_12/layer0/while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask28
6sequential_12/layer0/while/lstm_cell_1/strided_slice_2?
/sequential_12/layer0/while/lstm_cell_1/MatMul_6MatMul0sequential_12/layer0/while/lstm_cell_1/mul_6:z:0?sequential_12/layer0/while/lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????21
/sequential_12/layer0/while/lstm_cell_1/MatMul_6?
,sequential_12/layer0/while/lstm_cell_1/add_2AddV29sequential_12/layer0/while/lstm_cell_1/BiasAdd_2:output:09sequential_12/layer0/while/lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2.
,sequential_12/layer0/while/lstm_cell_1/add_2?
+sequential_12/layer0/while/lstm_cell_1/TanhTanh0sequential_12/layer0/while/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2-
+sequential_12/layer0/while/lstm_cell_1/Tanh?
,sequential_12/layer0/while/lstm_cell_1/mul_9Mul2sequential_12/layer0/while/lstm_cell_1/Sigmoid:y:0/sequential_12/layer0/while/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2.
,sequential_12/layer0/while/lstm_cell_1/mul_9?
,sequential_12/layer0/while/lstm_cell_1/add_3AddV20sequential_12/layer0/while/lstm_cell_1/mul_8:z:00sequential_12/layer0/while/lstm_cell_1/mul_9:z:0*
T0*(
_output_shapes
:??????????2.
,sequential_12/layer0/while/lstm_cell_1/add_3?
7sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_3ReadVariableOp@sequential_12_layer0_while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype029
7sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_3?
<sequential_12/layer0/while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2>
<sequential_12/layer0/while/lstm_cell_1/strided_slice_3/stack?
>sequential_12/layer0/while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2@
>sequential_12/layer0/while/lstm_cell_1/strided_slice_3/stack_1?
>sequential_12/layer0/while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2@
>sequential_12/layer0/while/lstm_cell_1/strided_slice_3/stack_2?
6sequential_12/layer0/while/lstm_cell_1/strided_slice_3StridedSlice?sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_3:value:0Esequential_12/layer0/while/lstm_cell_1/strided_slice_3/stack:output:0Gsequential_12/layer0/while/lstm_cell_1/strided_slice_3/stack_1:output:0Gsequential_12/layer0/while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask28
6sequential_12/layer0/while/lstm_cell_1/strided_slice_3?
/sequential_12/layer0/while/lstm_cell_1/MatMul_7MatMul0sequential_12/layer0/while/lstm_cell_1/mul_7:z:0?sequential_12/layer0/while/lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????21
/sequential_12/layer0/while/lstm_cell_1/MatMul_7?
,sequential_12/layer0/while/lstm_cell_1/add_4AddV29sequential_12/layer0/while/lstm_cell_1/BiasAdd_3:output:09sequential_12/layer0/while/lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2.
,sequential_12/layer0/while/lstm_cell_1/add_4?
0sequential_12/layer0/while/lstm_cell_1/Sigmoid_2Sigmoid0sequential_12/layer0/while/lstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:??????????22
0sequential_12/layer0/while/lstm_cell_1/Sigmoid_2?
-sequential_12/layer0/while/lstm_cell_1/Tanh_1Tanh0sequential_12/layer0/while/lstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:??????????2/
-sequential_12/layer0/while/lstm_cell_1/Tanh_1?
-sequential_12/layer0/while/lstm_cell_1/mul_10Mul4sequential_12/layer0/while/lstm_cell_1/Sigmoid_2:y:01sequential_12/layer0/while/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2/
-sequential_12/layer0/while/lstm_cell_1/mul_10?
?sequential_12/layer0/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_12_layer0_while_placeholder_1&sequential_12_layer0_while_placeholder1sequential_12/layer0/while/lstm_cell_1/mul_10:z:0*
_output_shapes
: *
element_dtype02A
?sequential_12/layer0/while/TensorArrayV2Write/TensorListSetItem?
 sequential_12/layer0/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_12/layer0/while/add/y?
sequential_12/layer0/while/addAddV2&sequential_12_layer0_while_placeholder)sequential_12/layer0/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_12/layer0/while/add?
"sequential_12/layer0/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_12/layer0/while/add_1/y?
 sequential_12/layer0/while/add_1AddV2Bsequential_12_layer0_while_sequential_12_layer0_while_loop_counter+sequential_12/layer0/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_12/layer0/while/add_1?
#sequential_12/layer0/while/IdentityIdentity$sequential_12/layer0/while/add_1:z:06^sequential_12/layer0/while/lstm_cell_1/ReadVariableOp8^sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_18^sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_28^sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_3<^sequential_12/layer0/while/lstm_cell_1/split/ReadVariableOp>^sequential_12/layer0/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2%
#sequential_12/layer0/while/Identity?
%sequential_12/layer0/while/Identity_1IdentityHsequential_12_layer0_while_sequential_12_layer0_while_maximum_iterations6^sequential_12/layer0/while/lstm_cell_1/ReadVariableOp8^sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_18^sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_28^sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_3<^sequential_12/layer0/while/lstm_cell_1/split/ReadVariableOp>^sequential_12/layer0/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2'
%sequential_12/layer0/while/Identity_1?
%sequential_12/layer0/while/Identity_2Identity"sequential_12/layer0/while/add:z:06^sequential_12/layer0/while/lstm_cell_1/ReadVariableOp8^sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_18^sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_28^sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_3<^sequential_12/layer0/while/lstm_cell_1/split/ReadVariableOp>^sequential_12/layer0/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2'
%sequential_12/layer0/while/Identity_2?
%sequential_12/layer0/while/Identity_3IdentityOsequential_12/layer0/while/TensorArrayV2Write/TensorListSetItem:output_handle:06^sequential_12/layer0/while/lstm_cell_1/ReadVariableOp8^sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_18^sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_28^sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_3<^sequential_12/layer0/while/lstm_cell_1/split/ReadVariableOp>^sequential_12/layer0/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2'
%sequential_12/layer0/while/Identity_3?
%sequential_12/layer0/while/Identity_4Identity1sequential_12/layer0/while/lstm_cell_1/mul_10:z:06^sequential_12/layer0/while/lstm_cell_1/ReadVariableOp8^sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_18^sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_28^sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_3<^sequential_12/layer0/while/lstm_cell_1/split/ReadVariableOp>^sequential_12/layer0/while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2'
%sequential_12/layer0/while/Identity_4?
%sequential_12/layer0/while/Identity_5Identity0sequential_12/layer0/while/lstm_cell_1/add_3:z:06^sequential_12/layer0/while/lstm_cell_1/ReadVariableOp8^sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_18^sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_28^sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_3<^sequential_12/layer0/while/lstm_cell_1/split/ReadVariableOp>^sequential_12/layer0/while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2'
%sequential_12/layer0/while/Identity_5"S
#sequential_12_layer0_while_identity,sequential_12/layer0/while/Identity:output:0"W
%sequential_12_layer0_while_identity_1.sequential_12/layer0/while/Identity_1:output:0"W
%sequential_12_layer0_while_identity_2.sequential_12/layer0/while/Identity_2:output:0"W
%sequential_12_layer0_while_identity_3.sequential_12/layer0/while/Identity_3:output:0"W
%sequential_12_layer0_while_identity_4.sequential_12/layer0/while/Identity_4:output:0"W
%sequential_12_layer0_while_identity_5.sequential_12/layer0/while/Identity_5:output:0"?
>sequential_12_layer0_while_lstm_cell_1_readvariableop_resource@sequential_12_layer0_while_lstm_cell_1_readvariableop_resource_0"?
Fsequential_12_layer0_while_lstm_cell_1_split_1_readvariableop_resourceHsequential_12_layer0_while_lstm_cell_1_split_1_readvariableop_resource_0"?
Dsequential_12_layer0_while_lstm_cell_1_split_readvariableop_resourceFsequential_12_layer0_while_lstm_cell_1_split_readvariableop_resource_0"?
?sequential_12_layer0_while_sequential_12_layer0_strided_slice_1Asequential_12_layer0_while_sequential_12_layer0_strided_slice_1_0"?
{sequential_12_layer0_while_tensorarrayv2read_tensorlistgetitem_sequential_12_layer0_tensorarrayunstack_tensorlistfromtensor}sequential_12_layer0_while_tensorarrayv2read_tensorlistgetitem_sequential_12_layer0_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2n
5sequential_12/layer0/while/lstm_cell_1/ReadVariableOp5sequential_12/layer0/while/lstm_cell_1/ReadVariableOp2r
7sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_17sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_12r
7sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_27sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_22r
7sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_37sequential_12/layer0/while/lstm_cell_1/ReadVariableOp_32z
;sequential_12/layer0/while/lstm_cell_1/split/ReadVariableOp;sequential_12/layer0/while/lstm_cell_1/split/ReadVariableOp2~
=sequential_12/layer0/while/lstm_cell_1/split_1/ReadVariableOp=sequential_12/layer0/while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
"__inference__traced_restore_351568
file_prefix$
 assignvariableop_dense_10_kernel$
 assignvariableop_1_dense_10_bias$
 assignvariableop_2_output_kernel"
assignvariableop_3_output_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate0
,assignvariableop_9_layer0_lstm_cell_1_kernel;
7assignvariableop_10_layer0_lstm_cell_1_recurrent_kernel/
+assignvariableop_11_layer0_lstm_cell_1_bias
assignvariableop_12_total
assignvariableop_13_count
assignvariableop_14_total_1
assignvariableop_15_count_1.
*assignvariableop_16_adam_dense_10_kernel_m,
(assignvariableop_17_adam_dense_10_bias_m,
(assignvariableop_18_adam_output_kernel_m*
&assignvariableop_19_adam_output_bias_m8
4assignvariableop_20_adam_layer0_lstm_cell_1_kernel_mB
>assignvariableop_21_adam_layer0_lstm_cell_1_recurrent_kernel_m6
2assignvariableop_22_adam_layer0_lstm_cell_1_bias_m.
*assignvariableop_23_adam_dense_10_kernel_v,
(assignvariableop_24_adam_dense_10_bias_v,
(assignvariableop_25_adam_output_kernel_v*
&assignvariableop_26_adam_output_bias_v8
4assignvariableop_27_adam_layer0_lstm_cell_1_kernel_vB
>assignvariableop_28_adam_layer0_lstm_cell_1_recurrent_kernel_v6
2assignvariableop_29_adam_layer0_lstm_cell_1_bias_v
identity_31??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_10_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_10_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp assignvariableop_2_output_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_output_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp,assignvariableop_9_layer0_lstm_cell_1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp7assignvariableop_10_layer0_lstm_cell_1_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp+assignvariableop_11_layer0_lstm_cell_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_dense_10_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_dense_10_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_output_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp&assignvariableop_19_adam_output_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp4assignvariableop_20_adam_layer0_lstm_cell_1_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp>assignvariableop_21_adam_layer0_lstm_cell_1_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp2assignvariableop_22_adam_layer0_lstm_cell_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_10_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_10_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_output_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_output_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_layer0_lstm_cell_1_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp>assignvariableop_28_adam_layer0_lstm_cell_1_recurrent_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp2assignvariableop_29_adam_layer0_lstm_cell_1_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_299
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_30?
Identity_31IdentityIdentity_30:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_31"#
identity_31Identity_31:output:0*?
_input_shapes|
z: ::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
while_cond_349911
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_349911___redundant_placeholder04
0while_while_cond_349911___redundant_placeholder14
0while_while_cond_349911___redundant_placeholder24
0while_while_cond_349911___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
while_body_349912
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
1while_lstm_cell_1_split_readvariableop_resource_07
3while_lstm_cell_1_split_1_readvariableop_resource_0/
+while_lstm_cell_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
/while_lstm_cell_1_split_readvariableop_resource5
1while_lstm_cell_1_split_1_readvariableop_resource-
)while_lstm_cell_1_readvariableop_resource?? while/lstm_cell_1/ReadVariableOp?"while/lstm_cell_1/ReadVariableOp_1?"while/lstm_cell_1/ReadVariableOp_2?"while/lstm_cell_1/ReadVariableOp_3?&while/lstm_cell_1/split/ReadVariableOp?(while/lstm_cell_1/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
!while/lstm_cell_1/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/ones_like/Shape?
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell_1/ones_like/Const?
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/ones_like?
while/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2!
while/lstm_cell_1/dropout/Const?
while/lstm_cell_1/dropout/MulMul$while/lstm_cell_1/ones_like:output:0(while/lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/dropout/Mul?
while/lstm_cell_1/dropout/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_1/dropout/Shape?
6while/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2???28
6while/lstm_cell_1/dropout/random_uniform/RandomUniform?
(while/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2*
(while/lstm_cell_1/dropout/GreaterEqual/y?
&while/lstm_cell_1/dropout/GreaterEqualGreaterEqual?while/lstm_cell_1/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2(
&while/lstm_cell_1/dropout/GreaterEqual?
while/lstm_cell_1/dropout/CastCast*while/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2 
while/lstm_cell_1/dropout/Cast?
while/lstm_cell_1/dropout/Mul_1Mul!while/lstm_cell_1/dropout/Mul:z:0"while/lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????P2!
while/lstm_cell_1/dropout/Mul_1?
!while/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!while/lstm_cell_1/dropout_1/Const?
while/lstm_cell_1/dropout_1/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????P2!
while/lstm_cell_1/dropout_1/Mul?
!while/lstm_cell_1/dropout_1/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_1/Shape?
8while/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2?܀2:
8while/lstm_cell_1/dropout_1/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_1/dropout_1/GreaterEqual/y?
(while/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2*
(while/lstm_cell_1/dropout_1/GreaterEqual?
 while/lstm_cell_1/dropout_1/CastCast,while/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2"
 while/lstm_cell_1/dropout_1/Cast?
!while/lstm_cell_1/dropout_1/Mul_1Mul#while/lstm_cell_1/dropout_1/Mul:z:0$while/lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????P2#
!while/lstm_cell_1/dropout_1/Mul_1?
!while/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!while/lstm_cell_1/dropout_2/Const?
while/lstm_cell_1/dropout_2/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????P2!
while/lstm_cell_1/dropout_2/Mul?
!while/lstm_cell_1/dropout_2/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_2/Shape?
8while/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_1/dropout_2/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_1/dropout_2/GreaterEqual/y?
(while/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2*
(while/lstm_cell_1/dropout_2/GreaterEqual?
 while/lstm_cell_1/dropout_2/CastCast,while/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2"
 while/lstm_cell_1/dropout_2/Cast?
!while/lstm_cell_1/dropout_2/Mul_1Mul#while/lstm_cell_1/dropout_2/Mul:z:0$while/lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????P2#
!while/lstm_cell_1/dropout_2/Mul_1?
!while/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!while/lstm_cell_1/dropout_3/Const?
while/lstm_cell_1/dropout_3/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????P2!
while/lstm_cell_1/dropout_3/Mul?
!while/lstm_cell_1/dropout_3/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_3/Shape?
8while/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2݄?2:
8while/lstm_cell_1/dropout_3/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_1/dropout_3/GreaterEqual/y?
(while/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2*
(while/lstm_cell_1/dropout_3/GreaterEqual?
 while/lstm_cell_1/dropout_3/CastCast,while/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2"
 while/lstm_cell_1/dropout_3/Cast?
!while/lstm_cell_1/dropout_3/Mul_1Mul#while/lstm_cell_1/dropout_3/Mul:z:0$while/lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????P2#
!while/lstm_cell_1/dropout_3/Mul_1?
#while/lstm_cell_1/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_1/ones_like_1/Shape?
#while/lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#while/lstm_cell_1/ones_like_1/Const?
while/lstm_cell_1/ones_like_1Fill,while/lstm_cell_1/ones_like_1/Shape:output:0,while/lstm_cell_1/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/ones_like_1?
!while/lstm_cell_1/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!while/lstm_cell_1/dropout_4/Const?
while/lstm_cell_1/dropout_4/MulMul&while/lstm_cell_1/ones_like_1:output:0*while/lstm_cell_1/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell_1/dropout_4/Mul?
!while/lstm_cell_1/dropout_4/ShapeShape&while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_4/Shape?
8while/lstm_cell_1/dropout_4/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_1/dropout_4/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_1/dropout_4/GreaterEqual/y?
(while/lstm_cell_1/dropout_4/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_4/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2*
(while/lstm_cell_1/dropout_4/GreaterEqual?
 while/lstm_cell_1/dropout_4/CastCast,while/lstm_cell_1/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2"
 while/lstm_cell_1/dropout_4/Cast?
!while/lstm_cell_1/dropout_4/Mul_1Mul#while/lstm_cell_1/dropout_4/Mul:z:0$while/lstm_cell_1/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2#
!while/lstm_cell_1/dropout_4/Mul_1?
!while/lstm_cell_1/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!while/lstm_cell_1/dropout_5/Const?
while/lstm_cell_1/dropout_5/MulMul&while/lstm_cell_1/ones_like_1:output:0*while/lstm_cell_1/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell_1/dropout_5/Mul?
!while/lstm_cell_1/dropout_5/ShapeShape&while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_5/Shape?
8while/lstm_cell_1/dropout_5/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_1/dropout_5/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_1/dropout_5/GreaterEqual/y?
(while/lstm_cell_1/dropout_5/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_5/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2*
(while/lstm_cell_1/dropout_5/GreaterEqual?
 while/lstm_cell_1/dropout_5/CastCast,while/lstm_cell_1/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2"
 while/lstm_cell_1/dropout_5/Cast?
!while/lstm_cell_1/dropout_5/Mul_1Mul#while/lstm_cell_1/dropout_5/Mul:z:0$while/lstm_cell_1/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2#
!while/lstm_cell_1/dropout_5/Mul_1?
!while/lstm_cell_1/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!while/lstm_cell_1/dropout_6/Const?
while/lstm_cell_1/dropout_6/MulMul&while/lstm_cell_1/ones_like_1:output:0*while/lstm_cell_1/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell_1/dropout_6/Mul?
!while/lstm_cell_1/dropout_6/ShapeShape&while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_6/Shape?
8while/lstm_cell_1/dropout_6/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??O2:
8while/lstm_cell_1/dropout_6/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_1/dropout_6/GreaterEqual/y?
(while/lstm_cell_1/dropout_6/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_6/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2*
(while/lstm_cell_1/dropout_6/GreaterEqual?
 while/lstm_cell_1/dropout_6/CastCast,while/lstm_cell_1/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2"
 while/lstm_cell_1/dropout_6/Cast?
!while/lstm_cell_1/dropout_6/Mul_1Mul#while/lstm_cell_1/dropout_6/Mul:z:0$while/lstm_cell_1/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2#
!while/lstm_cell_1/dropout_6/Mul_1?
!while/lstm_cell_1/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!while/lstm_cell_1/dropout_7/Const?
while/lstm_cell_1/dropout_7/MulMul&while/lstm_cell_1/ones_like_1:output:0*while/lstm_cell_1/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell_1/dropout_7/Mul?
!while/lstm_cell_1/dropout_7/ShapeShape&while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_7/Shape?
8while/lstm_cell_1/dropout_7/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_1/dropout_7/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_1/dropout_7/GreaterEqual/y?
(while/lstm_cell_1/dropout_7/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_7/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2*
(while/lstm_cell_1/dropout_7/GreaterEqual?
 while/lstm_cell_1/dropout_7/CastCast,while/lstm_cell_1/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2"
 while/lstm_cell_1/dropout_7/Cast?
!while/lstm_cell_1/dropout_7/Mul_1Mul#while/lstm_cell_1/dropout_7/Mul:z:0$while/lstm_cell_1/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2#
!while/lstm_cell_1/dropout_7/Mul_1?
while/lstm_cell_1/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/mul?
while/lstm_cell_1/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/mul_1?
while/lstm_cell_1/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/mul_2?
while/lstm_cell_1/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/mul_3t
while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const?
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_1/split/split_dim?
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes
:	P? *
dtype02(
&while/lstm_cell_1/split/ReadVariableOp?
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	P?:	P?:	P?:	P?*
	num_split2
while/lstm_cell_1/split?
while/lstm_cell_1/MatMulMatMulwhile/lstm_cell_1/mul:z:0 while/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul?
while/lstm_cell_1/MatMul_1MatMulwhile/lstm_cell_1/mul_1:z:0 while/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_1?
while/lstm_cell_1/MatMul_2MatMulwhile/lstm_cell_1/mul_2:z:0 while/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_2?
while/lstm_cell_1/MatMul_3MatMulwhile/lstm_cell_1/mul_3:z:0 while/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_3x
while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const_1?
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_1/split_1/split_dim?
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes	
:? *
dtype02*
(while/lstm_cell_1/split_1/ReadVariableOp?
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
while/lstm_cell_1/split_1?
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/BiasAdd?
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/BiasAdd_1?
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/BiasAdd_2?
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/BiasAdd_3?
while/lstm_cell_1/mul_4Mulwhile_placeholder_2%while/lstm_cell_1/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_4?
while/lstm_cell_1/mul_5Mulwhile_placeholder_2%while/lstm_cell_1/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_5?
while/lstm_cell_1/mul_6Mulwhile_placeholder_2%while/lstm_cell_1/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_6?
while/lstm_cell_1/mul_7Mulwhile_placeholder_2%while/lstm_cell_1/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_7?
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02"
 while/lstm_cell_1/ReadVariableOp?
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_1/strided_slice/stack?
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice/stack_1?
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_1/strided_slice/stack_2?
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell_1/strided_slice?
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul_4:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_4?
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add?
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Sigmoid?
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02$
"while/lstm_cell_1/ReadVariableOp_1?
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_1/stack?
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/lstm_cell_1/strided_slice_1/stack_1?
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_1/stack_2?
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_1?
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_5:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_5?
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add_1?
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Sigmoid_1?
while/lstm_cell_1/mul_8Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_8?
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02$
"while/lstm_cell_1/ReadVariableOp_2?
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_2/stack?
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/lstm_cell_1/strided_slice_2/stack_1?
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_2/stack_2?
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_2?
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_6:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_6?
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add_2?
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Tanh?
while/lstm_cell_1/mul_9Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_9?
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_8:z:0while/lstm_cell_1/mul_9:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add_3?
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02$
"while/lstm_cell_1/ReadVariableOp_3?
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_3/stack?
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_1/strided_slice_3/stack_1?
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_3/stack_2?
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_3?
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_7:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_7?
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add_4?
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Sigmoid_2?
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Tanh_1?
while/lstm_cell_1/mul_10Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_10?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_1/mul_10:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_1/add_3:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_348089
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_348089___redundant_placeholder04
0while_while_cond_348089___redundant_placeholder14
0while_while_cond_348089___redundant_placeholder24
0while_while_cond_348089___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
.__inference_sequential_12_layer_call_fn_348956
layer0_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_3489392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????P:::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????P
&
_user_specified_namelayer0_input
?
?
while_cond_350230
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_350230___redundant_placeholder04
0while_while_cond_350230___redundant_placeholder14
0while_while_cond_350230___redundant_placeholder24
0while_while_cond_350230___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
B__inference_layer0_layer_call_and_return_conditional_losses_350367
inputs_0-
)lstm_cell_1_split_readvariableop_resource/
+lstm_cell_1_split_1_readvariableop_resource'
#lstm_cell_1_readvariableop_resource
identity??lstm_cell_1/ReadVariableOp?lstm_cell_1/ReadVariableOp_1?lstm_cell_1/ReadVariableOp_2?lstm_cell_1/ReadVariableOp_3? lstm_cell_1/split/ReadVariableOp?"lstm_cell_1/split_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????P2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask2
strided_slice_2?
lstm_cell_1/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like/Shape
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_1/ones_like/Const?
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/ones_like|
lstm_cell_1/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like_1/Shape?
lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_1/ones_like_1/Const?
lstm_cell_1/ones_like_1Fill&lstm_cell_1/ones_like_1/Shape:output:0&lstm_cell_1/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/ones_like_1?
lstm_cell_1/mulMulstrided_slice_2:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/mul?
lstm_cell_1/mul_1Mulstrided_slice_2:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/mul_1?
lstm_cell_1/mul_2Mulstrided_slice_2:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/mul_2?
lstm_cell_1/mul_3Mulstrided_slice_2:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/mul_3h
lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dim?
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource*
_output_shapes
:	P? *
dtype02"
 lstm_cell_1/split/ReadVariableOp?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	P?:	P?:	P?:	P?*
	num_split2
lstm_cell_1/split?
lstm_cell_1/MatMulMatMullstm_cell_1/mul:z:0lstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul?
lstm_cell_1/MatMul_1MatMullstm_cell_1/mul_1:z:0lstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_1?
lstm_cell_1/MatMul_2MatMullstm_cell_1/mul_2:z:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_2?
lstm_cell_1/MatMul_3MatMullstm_cell_1/mul_3:z:0lstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_3l
lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const_1?
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_1/split_1/split_dim?
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes	
:? *
dtype02$
"lstm_cell_1/split_1/ReadVariableOp?
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell_1/split_1?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/BiasAdd?
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_1/BiasAdd_1?
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_1/BiasAdd_2?
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_1/BiasAdd_3?
lstm_cell_1/mul_4Mulzeros:output:0 lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_4?
lstm_cell_1/mul_5Mulzeros:output:0 lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_5?
lstm_cell_1/mul_6Mulzeros:output:0 lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_6?
lstm_cell_1/mul_7Mulzeros:output:0 lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_7?
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
lstm_cell_1/ReadVariableOp?
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_1/strided_slice/stack?
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice/stack_1?
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_1/strided_slice/stack_2?
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_1/strided_slice?
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul_4:z:0"lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_4?
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add}
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Sigmoid?
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
lstm_cell_1/ReadVariableOp_1?
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_1/stack?
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#lstm_cell_1/strided_slice_1/stack_1?
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_1/stack_2?
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_1?
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_5:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_5?
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add_1?
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Sigmoid_1?
lstm_cell_1/mul_8Mullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_8?
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
lstm_cell_1/ReadVariableOp_2?
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_2/stack?
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#lstm_cell_1/strided_slice_2/stack_1?
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_2/stack_2?
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_2?
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_6:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_6?
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add_2v
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Tanh?
lstm_cell_1/mul_9Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_9?
lstm_cell_1/add_3AddV2lstm_cell_1/mul_8:z:0lstm_cell_1/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add_3?
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
lstm_cell_1/ReadVariableOp_3?
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_3/stack?
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_1/strided_slice_3/stack_1?
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_3/stack_2?
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_3?
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_7:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_7?
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add_4?
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Sigmoid_2z
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Tanh_1?
lstm_cell_1/mul_10Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_350231*
condR
while_cond_350230*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????P:::28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????P
"
_user_specified_name
inputs/0
?N
?
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_347664

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpX
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????P2
	ones_like\
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
ones_like_1_
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:?????????P2
mulc
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????P2
mul_1c
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????P2
mul_2c
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????P2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	P? *
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	P?:	P?:	P?:	P?*
	num_split2
splitf
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:??????????2
MatMull
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:??????????2

MatMul_1l
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:??????????2

MatMul_2l
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:??????????2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:? *
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????2
	BiasAdd_3f
mul_4Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:??????????2
mul_4f
mul_5Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:??????????2
mul_5f
mul_6Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:??????????2
mul_6f
mul_7Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:??????????2
mul_7z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
?? *
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slicet
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
?? *
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_1v
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1a
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mul_8~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
?? *
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_2v
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????2
Tanh_
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????2
mul_9`
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:??????????2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
?? *
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_3v
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:??????????2
Tanh_1e
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
mul_10?
IdentityIdentity
mul_10:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity
mul_10:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????P:??????????:??????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
??
?

layer0_while_body_349208*
&layer0_while_layer0_while_loop_counter0
,layer0_while_layer0_while_maximum_iterations
layer0_while_placeholder
layer0_while_placeholder_1
layer0_while_placeholder_2
layer0_while_placeholder_3)
%layer0_while_layer0_strided_slice_1_0e
alayer0_while_tensorarrayv2read_tensorlistgetitem_layer0_tensorarrayunstack_tensorlistfromtensor_0<
8layer0_while_lstm_cell_1_split_readvariableop_resource_0>
:layer0_while_lstm_cell_1_split_1_readvariableop_resource_06
2layer0_while_lstm_cell_1_readvariableop_resource_0
layer0_while_identity
layer0_while_identity_1
layer0_while_identity_2
layer0_while_identity_3
layer0_while_identity_4
layer0_while_identity_5'
#layer0_while_layer0_strided_slice_1c
_layer0_while_tensorarrayv2read_tensorlistgetitem_layer0_tensorarrayunstack_tensorlistfromtensor:
6layer0_while_lstm_cell_1_split_readvariableop_resource<
8layer0_while_lstm_cell_1_split_1_readvariableop_resource4
0layer0_while_lstm_cell_1_readvariableop_resource??'layer0/while/lstm_cell_1/ReadVariableOp?)layer0/while/lstm_cell_1/ReadVariableOp_1?)layer0/while/lstm_cell_1/ReadVariableOp_2?)layer0/while/lstm_cell_1/ReadVariableOp_3?-layer0/while/lstm_cell_1/split/ReadVariableOp?/layer0/while/lstm_cell_1/split_1/ReadVariableOp?
>layer0/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   2@
>layer0/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0layer0/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalayer0_while_tensorarrayv2read_tensorlistgetitem_layer0_tensorarrayunstack_tensorlistfromtensor_0layer0_while_placeholderGlayer0/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype022
0layer0/while/TensorArrayV2Read/TensorListGetItem?
(layer0/while/lstm_cell_1/ones_like/ShapeShape7layer0/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2*
(layer0/while/lstm_cell_1/ones_like/Shape?
(layer0/while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(layer0/while/lstm_cell_1/ones_like/Const?
"layer0/while/lstm_cell_1/ones_likeFill1layer0/while/lstm_cell_1/ones_like/Shape:output:01layer0/while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????P2$
"layer0/while/lstm_cell_1/ones_like?
&layer0/while/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2(
&layer0/while/lstm_cell_1/dropout/Const?
$layer0/while/lstm_cell_1/dropout/MulMul+layer0/while/lstm_cell_1/ones_like:output:0/layer0/while/lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????P2&
$layer0/while/lstm_cell_1/dropout/Mul?
&layer0/while/lstm_cell_1/dropout/ShapeShape+layer0/while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2(
&layer0/while/lstm_cell_1/dropout/Shape?
=layer0/while/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform/layer0/while/lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2??2?
=layer0/while/lstm_cell_1/dropout/random_uniform/RandomUniform?
/layer0/while/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>21
/layer0/while/lstm_cell_1/dropout/GreaterEqual/y?
-layer0/while/lstm_cell_1/dropout/GreaterEqualGreaterEqualFlayer0/while/lstm_cell_1/dropout/random_uniform/RandomUniform:output:08layer0/while/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2/
-layer0/while/lstm_cell_1/dropout/GreaterEqual?
%layer0/while/lstm_cell_1/dropout/CastCast1layer0/while/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2'
%layer0/while/lstm_cell_1/dropout/Cast?
&layer0/while/lstm_cell_1/dropout/Mul_1Mul(layer0/while/lstm_cell_1/dropout/Mul:z:0)layer0/while/lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????P2(
&layer0/while/lstm_cell_1/dropout/Mul_1?
(layer0/while/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2*
(layer0/while/lstm_cell_1/dropout_1/Const?
&layer0/while/lstm_cell_1/dropout_1/MulMul+layer0/while/lstm_cell_1/ones_like:output:01layer0/while/lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????P2(
&layer0/while/lstm_cell_1/dropout_1/Mul?
(layer0/while/lstm_cell_1/dropout_1/ShapeShape+layer0/while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2*
(layer0/while/lstm_cell_1/dropout_1/Shape?
?layer0/while/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform1layer0/while/lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2???2A
?layer0/while/lstm_cell_1/dropout_1/random_uniform/RandomUniform?
1layer0/while/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>23
1layer0/while/lstm_cell_1/dropout_1/GreaterEqual/y?
/layer0/while/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualHlayer0/while/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0:layer0/while/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P21
/layer0/while/lstm_cell_1/dropout_1/GreaterEqual?
'layer0/while/lstm_cell_1/dropout_1/CastCast3layer0/while/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2)
'layer0/while/lstm_cell_1/dropout_1/Cast?
(layer0/while/lstm_cell_1/dropout_1/Mul_1Mul*layer0/while/lstm_cell_1/dropout_1/Mul:z:0+layer0/while/lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????P2*
(layer0/while/lstm_cell_1/dropout_1/Mul_1?
(layer0/while/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2*
(layer0/while/lstm_cell_1/dropout_2/Const?
&layer0/while/lstm_cell_1/dropout_2/MulMul+layer0/while/lstm_cell_1/ones_like:output:01layer0/while/lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????P2(
&layer0/while/lstm_cell_1/dropout_2/Mul?
(layer0/while/lstm_cell_1/dropout_2/ShapeShape+layer0/while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2*
(layer0/while/lstm_cell_1/dropout_2/Shape?
?layer0/while/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform1layer0/while/lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2???2A
?layer0/while/lstm_cell_1/dropout_2/random_uniform/RandomUniform?
1layer0/while/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>23
1layer0/while/lstm_cell_1/dropout_2/GreaterEqual/y?
/layer0/while/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualHlayer0/while/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0:layer0/while/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P21
/layer0/while/lstm_cell_1/dropout_2/GreaterEqual?
'layer0/while/lstm_cell_1/dropout_2/CastCast3layer0/while/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2)
'layer0/while/lstm_cell_1/dropout_2/Cast?
(layer0/while/lstm_cell_1/dropout_2/Mul_1Mul*layer0/while/lstm_cell_1/dropout_2/Mul:z:0+layer0/while/lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????P2*
(layer0/while/lstm_cell_1/dropout_2/Mul_1?
(layer0/while/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2*
(layer0/while/lstm_cell_1/dropout_3/Const?
&layer0/while/lstm_cell_1/dropout_3/MulMul+layer0/while/lstm_cell_1/ones_like:output:01layer0/while/lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????P2(
&layer0/while/lstm_cell_1/dropout_3/Mul?
(layer0/while/lstm_cell_1/dropout_3/ShapeShape+layer0/while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2*
(layer0/while/lstm_cell_1/dropout_3/Shape?
?layer0/while/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform1layer0/while/lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2???2A
?layer0/while/lstm_cell_1/dropout_3/random_uniform/RandomUniform?
1layer0/while/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>23
1layer0/while/lstm_cell_1/dropout_3/GreaterEqual/y?
/layer0/while/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualHlayer0/while/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0:layer0/while/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P21
/layer0/while/lstm_cell_1/dropout_3/GreaterEqual?
'layer0/while/lstm_cell_1/dropout_3/CastCast3layer0/while/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2)
'layer0/while/lstm_cell_1/dropout_3/Cast?
(layer0/while/lstm_cell_1/dropout_3/Mul_1Mul*layer0/while/lstm_cell_1/dropout_3/Mul:z:0+layer0/while/lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????P2*
(layer0/while/lstm_cell_1/dropout_3/Mul_1?
*layer0/while/lstm_cell_1/ones_like_1/ShapeShapelayer0_while_placeholder_2*
T0*
_output_shapes
:2,
*layer0/while/lstm_cell_1/ones_like_1/Shape?
*layer0/while/lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*layer0/while/lstm_cell_1/ones_like_1/Const?
$layer0/while/lstm_cell_1/ones_like_1Fill3layer0/while/lstm_cell_1/ones_like_1/Shape:output:03layer0/while/lstm_cell_1/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2&
$layer0/while/lstm_cell_1/ones_like_1?
(layer0/while/lstm_cell_1/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2*
(layer0/while/lstm_cell_1/dropout_4/Const?
&layer0/while/lstm_cell_1/dropout_4/MulMul-layer0/while/lstm_cell_1/ones_like_1:output:01layer0/while/lstm_cell_1/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2(
&layer0/while/lstm_cell_1/dropout_4/Mul?
(layer0/while/lstm_cell_1/dropout_4/ShapeShape-layer0/while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2*
(layer0/while/lstm_cell_1/dropout_4/Shape?
?layer0/while/lstm_cell_1/dropout_4/random_uniform/RandomUniformRandomUniform1layer0/while/lstm_cell_1/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2A
?layer0/while/lstm_cell_1/dropout_4/random_uniform/RandomUniform?
1layer0/while/lstm_cell_1/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>23
1layer0/while/lstm_cell_1/dropout_4/GreaterEqual/y?
/layer0/while/lstm_cell_1/dropout_4/GreaterEqualGreaterEqualHlayer0/while/lstm_cell_1/dropout_4/random_uniform/RandomUniform:output:0:layer0/while/lstm_cell_1/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????21
/layer0/while/lstm_cell_1/dropout_4/GreaterEqual?
'layer0/while/lstm_cell_1/dropout_4/CastCast3layer0/while/lstm_cell_1/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2)
'layer0/while/lstm_cell_1/dropout_4/Cast?
(layer0/while/lstm_cell_1/dropout_4/Mul_1Mul*layer0/while/lstm_cell_1/dropout_4/Mul:z:0+layer0/while/lstm_cell_1/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2*
(layer0/while/lstm_cell_1/dropout_4/Mul_1?
(layer0/while/lstm_cell_1/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2*
(layer0/while/lstm_cell_1/dropout_5/Const?
&layer0/while/lstm_cell_1/dropout_5/MulMul-layer0/while/lstm_cell_1/ones_like_1:output:01layer0/while/lstm_cell_1/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2(
&layer0/while/lstm_cell_1/dropout_5/Mul?
(layer0/while/lstm_cell_1/dropout_5/ShapeShape-layer0/while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2*
(layer0/while/lstm_cell_1/dropout_5/Shape?
?layer0/while/lstm_cell_1/dropout_5/random_uniform/RandomUniformRandomUniform1layer0/while/lstm_cell_1/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2A
?layer0/while/lstm_cell_1/dropout_5/random_uniform/RandomUniform?
1layer0/while/lstm_cell_1/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>23
1layer0/while/lstm_cell_1/dropout_5/GreaterEqual/y?
/layer0/while/lstm_cell_1/dropout_5/GreaterEqualGreaterEqualHlayer0/while/lstm_cell_1/dropout_5/random_uniform/RandomUniform:output:0:layer0/while/lstm_cell_1/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????21
/layer0/while/lstm_cell_1/dropout_5/GreaterEqual?
'layer0/while/lstm_cell_1/dropout_5/CastCast3layer0/while/lstm_cell_1/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2)
'layer0/while/lstm_cell_1/dropout_5/Cast?
(layer0/while/lstm_cell_1/dropout_5/Mul_1Mul*layer0/while/lstm_cell_1/dropout_5/Mul:z:0+layer0/while/lstm_cell_1/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2*
(layer0/while/lstm_cell_1/dropout_5/Mul_1?
(layer0/while/lstm_cell_1/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2*
(layer0/while/lstm_cell_1/dropout_6/Const?
&layer0/while/lstm_cell_1/dropout_6/MulMul-layer0/while/lstm_cell_1/ones_like_1:output:01layer0/while/lstm_cell_1/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2(
&layer0/while/lstm_cell_1/dropout_6/Mul?
(layer0/while/lstm_cell_1/dropout_6/ShapeShape-layer0/while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2*
(layer0/while/lstm_cell_1/dropout_6/Shape?
?layer0/while/lstm_cell_1/dropout_6/random_uniform/RandomUniformRandomUniform1layer0/while/lstm_cell_1/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2ᘹ2A
?layer0/while/lstm_cell_1/dropout_6/random_uniform/RandomUniform?
1layer0/while/lstm_cell_1/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>23
1layer0/while/lstm_cell_1/dropout_6/GreaterEqual/y?
/layer0/while/lstm_cell_1/dropout_6/GreaterEqualGreaterEqualHlayer0/while/lstm_cell_1/dropout_6/random_uniform/RandomUniform:output:0:layer0/while/lstm_cell_1/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????21
/layer0/while/lstm_cell_1/dropout_6/GreaterEqual?
'layer0/while/lstm_cell_1/dropout_6/CastCast3layer0/while/lstm_cell_1/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2)
'layer0/while/lstm_cell_1/dropout_6/Cast?
(layer0/while/lstm_cell_1/dropout_6/Mul_1Mul*layer0/while/lstm_cell_1/dropout_6/Mul:z:0+layer0/while/lstm_cell_1/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2*
(layer0/while/lstm_cell_1/dropout_6/Mul_1?
(layer0/while/lstm_cell_1/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2*
(layer0/while/lstm_cell_1/dropout_7/Const?
&layer0/while/lstm_cell_1/dropout_7/MulMul-layer0/while/lstm_cell_1/ones_like_1:output:01layer0/while/lstm_cell_1/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2(
&layer0/while/lstm_cell_1/dropout_7/Mul?
(layer0/while/lstm_cell_1/dropout_7/ShapeShape-layer0/while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2*
(layer0/while/lstm_cell_1/dropout_7/Shape?
?layer0/while/lstm_cell_1/dropout_7/random_uniform/RandomUniformRandomUniform1layer0/while/lstm_cell_1/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2A
?layer0/while/lstm_cell_1/dropout_7/random_uniform/RandomUniform?
1layer0/while/lstm_cell_1/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>23
1layer0/while/lstm_cell_1/dropout_7/GreaterEqual/y?
/layer0/while/lstm_cell_1/dropout_7/GreaterEqualGreaterEqualHlayer0/while/lstm_cell_1/dropout_7/random_uniform/RandomUniform:output:0:layer0/while/lstm_cell_1/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????21
/layer0/while/lstm_cell_1/dropout_7/GreaterEqual?
'layer0/while/lstm_cell_1/dropout_7/CastCast3layer0/while/lstm_cell_1/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2)
'layer0/while/lstm_cell_1/dropout_7/Cast?
(layer0/while/lstm_cell_1/dropout_7/Mul_1Mul*layer0/while/lstm_cell_1/dropout_7/Mul:z:0+layer0/while/lstm_cell_1/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2*
(layer0/while/lstm_cell_1/dropout_7/Mul_1?
layer0/while/lstm_cell_1/mulMul7layer0/while/TensorArrayV2Read/TensorListGetItem:item:0*layer0/while/lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
layer0/while/lstm_cell_1/mul?
layer0/while/lstm_cell_1/mul_1Mul7layer0/while/TensorArrayV2Read/TensorListGetItem:item:0,layer0/while/lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2 
layer0/while/lstm_cell_1/mul_1?
layer0/while/lstm_cell_1/mul_2Mul7layer0/while/TensorArrayV2Read/TensorListGetItem:item:0,layer0/while/lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2 
layer0/while/lstm_cell_1/mul_2?
layer0/while/lstm_cell_1/mul_3Mul7layer0/while/TensorArrayV2Read/TensorListGetItem:item:0,layer0/while/lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2 
layer0/while/lstm_cell_1/mul_3?
layer0/while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
layer0/while/lstm_cell_1/Const?
(layer0/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(layer0/while/lstm_cell_1/split/split_dim?
-layer0/while/lstm_cell_1/split/ReadVariableOpReadVariableOp8layer0_while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes
:	P? *
dtype02/
-layer0/while/lstm_cell_1/split/ReadVariableOp?
layer0/while/lstm_cell_1/splitSplit1layer0/while/lstm_cell_1/split/split_dim:output:05layer0/while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	P?:	P?:	P?:	P?*
	num_split2 
layer0/while/lstm_cell_1/split?
layer0/while/lstm_cell_1/MatMulMatMul layer0/while/lstm_cell_1/mul:z:0'layer0/while/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????2!
layer0/while/lstm_cell_1/MatMul?
!layer0/while/lstm_cell_1/MatMul_1MatMul"layer0/while/lstm_cell_1/mul_1:z:0'layer0/while/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????2#
!layer0/while/lstm_cell_1/MatMul_1?
!layer0/while/lstm_cell_1/MatMul_2MatMul"layer0/while/lstm_cell_1/mul_2:z:0'layer0/while/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????2#
!layer0/while/lstm_cell_1/MatMul_2?
!layer0/while/lstm_cell_1/MatMul_3MatMul"layer0/while/lstm_cell_1/mul_3:z:0'layer0/while/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????2#
!layer0/while/lstm_cell_1/MatMul_3?
 layer0/while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2"
 layer0/while/lstm_cell_1/Const_1?
*layer0/while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*layer0/while/lstm_cell_1/split_1/split_dim?
/layer0/while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp:layer0_while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes	
:? *
dtype021
/layer0/while/lstm_cell_1/split_1/ReadVariableOp?
 layer0/while/lstm_cell_1/split_1Split3layer0/while/lstm_cell_1/split_1/split_dim:output:07layer0/while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2"
 layer0/while/lstm_cell_1/split_1?
 layer0/while/lstm_cell_1/BiasAddBiasAdd)layer0/while/lstm_cell_1/MatMul:product:0)layer0/while/lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:??????????2"
 layer0/while/lstm_cell_1/BiasAdd?
"layer0/while/lstm_cell_1/BiasAdd_1BiasAdd+layer0/while/lstm_cell_1/MatMul_1:product:0)layer0/while/lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:??????????2$
"layer0/while/lstm_cell_1/BiasAdd_1?
"layer0/while/lstm_cell_1/BiasAdd_2BiasAdd+layer0/while/lstm_cell_1/MatMul_2:product:0)layer0/while/lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:??????????2$
"layer0/while/lstm_cell_1/BiasAdd_2?
"layer0/while/lstm_cell_1/BiasAdd_3BiasAdd+layer0/while/lstm_cell_1/MatMul_3:product:0)layer0/while/lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:??????????2$
"layer0/while/lstm_cell_1/BiasAdd_3?
layer0/while/lstm_cell_1/mul_4Mullayer0_while_placeholder_2,layer0/while/lstm_cell_1/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2 
layer0/while/lstm_cell_1/mul_4?
layer0/while/lstm_cell_1/mul_5Mullayer0_while_placeholder_2,layer0/while/lstm_cell_1/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2 
layer0/while/lstm_cell_1/mul_5?
layer0/while/lstm_cell_1/mul_6Mullayer0_while_placeholder_2,layer0/while/lstm_cell_1/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2 
layer0/while/lstm_cell_1/mul_6?
layer0/while/lstm_cell_1/mul_7Mullayer0_while_placeholder_2,layer0/while/lstm_cell_1/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2 
layer0/while/lstm_cell_1/mul_7?
'layer0/while/lstm_cell_1/ReadVariableOpReadVariableOp2layer0_while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02)
'layer0/while/lstm_cell_1/ReadVariableOp?
,layer0/while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,layer0/while/lstm_cell_1/strided_slice/stack?
.layer0/while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.layer0/while/lstm_cell_1/strided_slice/stack_1?
.layer0/while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.layer0/while/lstm_cell_1/strided_slice/stack_2?
&layer0/while/lstm_cell_1/strided_sliceStridedSlice/layer0/while/lstm_cell_1/ReadVariableOp:value:05layer0/while/lstm_cell_1/strided_slice/stack:output:07layer0/while/lstm_cell_1/strided_slice/stack_1:output:07layer0/while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&layer0/while/lstm_cell_1/strided_slice?
!layer0/while/lstm_cell_1/MatMul_4MatMul"layer0/while/lstm_cell_1/mul_4:z:0/layer0/while/lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2#
!layer0/while/lstm_cell_1/MatMul_4?
layer0/while/lstm_cell_1/addAddV2)layer0/while/lstm_cell_1/BiasAdd:output:0+layer0/while/lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
layer0/while/lstm_cell_1/add?
 layer0/while/lstm_cell_1/SigmoidSigmoid layer0/while/lstm_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2"
 layer0/while/lstm_cell_1/Sigmoid?
)layer0/while/lstm_cell_1/ReadVariableOp_1ReadVariableOp2layer0_while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02+
)layer0/while/lstm_cell_1/ReadVariableOp_1?
.layer0/while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.layer0/while/lstm_cell_1/strided_slice_1/stack?
0layer0/while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0layer0/while/lstm_cell_1/strided_slice_1/stack_1?
0layer0/while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0layer0/while/lstm_cell_1/strided_slice_1/stack_2?
(layer0/while/lstm_cell_1/strided_slice_1StridedSlice1layer0/while/lstm_cell_1/ReadVariableOp_1:value:07layer0/while/lstm_cell_1/strided_slice_1/stack:output:09layer0/while/lstm_cell_1/strided_slice_1/stack_1:output:09layer0/while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2*
(layer0/while/lstm_cell_1/strided_slice_1?
!layer0/while/lstm_cell_1/MatMul_5MatMul"layer0/while/lstm_cell_1/mul_5:z:01layer0/while/lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2#
!layer0/while/lstm_cell_1/MatMul_5?
layer0/while/lstm_cell_1/add_1AddV2+layer0/while/lstm_cell_1/BiasAdd_1:output:0+layer0/while/lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2 
layer0/while/lstm_cell_1/add_1?
"layer0/while/lstm_cell_1/Sigmoid_1Sigmoid"layer0/while/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2$
"layer0/while/lstm_cell_1/Sigmoid_1?
layer0/while/lstm_cell_1/mul_8Mul&layer0/while/lstm_cell_1/Sigmoid_1:y:0layer0_while_placeholder_3*
T0*(
_output_shapes
:??????????2 
layer0/while/lstm_cell_1/mul_8?
)layer0/while/lstm_cell_1/ReadVariableOp_2ReadVariableOp2layer0_while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02+
)layer0/while/lstm_cell_1/ReadVariableOp_2?
.layer0/while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.layer0/while/lstm_cell_1/strided_slice_2/stack?
0layer0/while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0layer0/while/lstm_cell_1/strided_slice_2/stack_1?
0layer0/while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0layer0/while/lstm_cell_1/strided_slice_2/stack_2?
(layer0/while/lstm_cell_1/strided_slice_2StridedSlice1layer0/while/lstm_cell_1/ReadVariableOp_2:value:07layer0/while/lstm_cell_1/strided_slice_2/stack:output:09layer0/while/lstm_cell_1/strided_slice_2/stack_1:output:09layer0/while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2*
(layer0/while/lstm_cell_1/strided_slice_2?
!layer0/while/lstm_cell_1/MatMul_6MatMul"layer0/while/lstm_cell_1/mul_6:z:01layer0/while/lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2#
!layer0/while/lstm_cell_1/MatMul_6?
layer0/while/lstm_cell_1/add_2AddV2+layer0/while/lstm_cell_1/BiasAdd_2:output:0+layer0/while/lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2 
layer0/while/lstm_cell_1/add_2?
layer0/while/lstm_cell_1/TanhTanh"layer0/while/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
layer0/while/lstm_cell_1/Tanh?
layer0/while/lstm_cell_1/mul_9Mul$layer0/while/lstm_cell_1/Sigmoid:y:0!layer0/while/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2 
layer0/while/lstm_cell_1/mul_9?
layer0/while/lstm_cell_1/add_3AddV2"layer0/while/lstm_cell_1/mul_8:z:0"layer0/while/lstm_cell_1/mul_9:z:0*
T0*(
_output_shapes
:??????????2 
layer0/while/lstm_cell_1/add_3?
)layer0/while/lstm_cell_1/ReadVariableOp_3ReadVariableOp2layer0_while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02+
)layer0/while/lstm_cell_1/ReadVariableOp_3?
.layer0/while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.layer0/while/lstm_cell_1/strided_slice_3/stack?
0layer0/while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0layer0/while/lstm_cell_1/strided_slice_3/stack_1?
0layer0/while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0layer0/while/lstm_cell_1/strided_slice_3/stack_2?
(layer0/while/lstm_cell_1/strided_slice_3StridedSlice1layer0/while/lstm_cell_1/ReadVariableOp_3:value:07layer0/while/lstm_cell_1/strided_slice_3/stack:output:09layer0/while/lstm_cell_1/strided_slice_3/stack_1:output:09layer0/while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2*
(layer0/while/lstm_cell_1/strided_slice_3?
!layer0/while/lstm_cell_1/MatMul_7MatMul"layer0/while/lstm_cell_1/mul_7:z:01layer0/while/lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2#
!layer0/while/lstm_cell_1/MatMul_7?
layer0/while/lstm_cell_1/add_4AddV2+layer0/while/lstm_cell_1/BiasAdd_3:output:0+layer0/while/lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2 
layer0/while/lstm_cell_1/add_4?
"layer0/while/lstm_cell_1/Sigmoid_2Sigmoid"layer0/while/lstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:??????????2$
"layer0/while/lstm_cell_1/Sigmoid_2?
layer0/while/lstm_cell_1/Tanh_1Tanh"layer0/while/lstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:??????????2!
layer0/while/lstm_cell_1/Tanh_1?
layer0/while/lstm_cell_1/mul_10Mul&layer0/while/lstm_cell_1/Sigmoid_2:y:0#layer0/while/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2!
layer0/while/lstm_cell_1/mul_10?
1layer0/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlayer0_while_placeholder_1layer0_while_placeholder#layer0/while/lstm_cell_1/mul_10:z:0*
_output_shapes
: *
element_dtype023
1layer0/while/TensorArrayV2Write/TensorListSetItemj
layer0/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
layer0/while/add/y?
layer0/while/addAddV2layer0_while_placeholderlayer0/while/add/y:output:0*
T0*
_output_shapes
: 2
layer0/while/addn
layer0/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
layer0/while/add_1/y?
layer0/while/add_1AddV2&layer0_while_layer0_while_loop_counterlayer0/while/add_1/y:output:0*
T0*
_output_shapes
: 2
layer0/while/add_1?
layer0/while/IdentityIdentitylayer0/while/add_1:z:0(^layer0/while/lstm_cell_1/ReadVariableOp*^layer0/while/lstm_cell_1/ReadVariableOp_1*^layer0/while/lstm_cell_1/ReadVariableOp_2*^layer0/while/lstm_cell_1/ReadVariableOp_3.^layer0/while/lstm_cell_1/split/ReadVariableOp0^layer0/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
layer0/while/Identity?
layer0/while/Identity_1Identity,layer0_while_layer0_while_maximum_iterations(^layer0/while/lstm_cell_1/ReadVariableOp*^layer0/while/lstm_cell_1/ReadVariableOp_1*^layer0/while/lstm_cell_1/ReadVariableOp_2*^layer0/while/lstm_cell_1/ReadVariableOp_3.^layer0/while/lstm_cell_1/split/ReadVariableOp0^layer0/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
layer0/while/Identity_1?
layer0/while/Identity_2Identitylayer0/while/add:z:0(^layer0/while/lstm_cell_1/ReadVariableOp*^layer0/while/lstm_cell_1/ReadVariableOp_1*^layer0/while/lstm_cell_1/ReadVariableOp_2*^layer0/while/lstm_cell_1/ReadVariableOp_3.^layer0/while/lstm_cell_1/split/ReadVariableOp0^layer0/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
layer0/while/Identity_2?
layer0/while/Identity_3IdentityAlayer0/while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^layer0/while/lstm_cell_1/ReadVariableOp*^layer0/while/lstm_cell_1/ReadVariableOp_1*^layer0/while/lstm_cell_1/ReadVariableOp_2*^layer0/while/lstm_cell_1/ReadVariableOp_3.^layer0/while/lstm_cell_1/split/ReadVariableOp0^layer0/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
layer0/while/Identity_3?
layer0/while/Identity_4Identity#layer0/while/lstm_cell_1/mul_10:z:0(^layer0/while/lstm_cell_1/ReadVariableOp*^layer0/while/lstm_cell_1/ReadVariableOp_1*^layer0/while/lstm_cell_1/ReadVariableOp_2*^layer0/while/lstm_cell_1/ReadVariableOp_3.^layer0/while/lstm_cell_1/split/ReadVariableOp0^layer0/while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
layer0/while/Identity_4?
layer0/while/Identity_5Identity"layer0/while/lstm_cell_1/add_3:z:0(^layer0/while/lstm_cell_1/ReadVariableOp*^layer0/while/lstm_cell_1/ReadVariableOp_1*^layer0/while/lstm_cell_1/ReadVariableOp_2*^layer0/while/lstm_cell_1/ReadVariableOp_3.^layer0/while/lstm_cell_1/split/ReadVariableOp0^layer0/while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
layer0/while/Identity_5"7
layer0_while_identitylayer0/while/Identity:output:0";
layer0_while_identity_1 layer0/while/Identity_1:output:0";
layer0_while_identity_2 layer0/while/Identity_2:output:0";
layer0_while_identity_3 layer0/while/Identity_3:output:0";
layer0_while_identity_4 layer0/while/Identity_4:output:0";
layer0_while_identity_5 layer0/while/Identity_5:output:0"L
#layer0_while_layer0_strided_slice_1%layer0_while_layer0_strided_slice_1_0"f
0layer0_while_lstm_cell_1_readvariableop_resource2layer0_while_lstm_cell_1_readvariableop_resource_0"v
8layer0_while_lstm_cell_1_split_1_readvariableop_resource:layer0_while_lstm_cell_1_split_1_readvariableop_resource_0"r
6layer0_while_lstm_cell_1_split_readvariableop_resource8layer0_while_lstm_cell_1_split_readvariableop_resource_0"?
_layer0_while_tensorarrayv2read_tensorlistgetitem_layer0_tensorarrayunstack_tensorlistfromtensoralayer0_while_tensorarrayv2read_tensorlistgetitem_layer0_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2R
'layer0/while/lstm_cell_1/ReadVariableOp'layer0/while/lstm_cell_1/ReadVariableOp2V
)layer0/while/lstm_cell_1/ReadVariableOp_1)layer0/while/lstm_cell_1/ReadVariableOp_12V
)layer0/while/lstm_cell_1/ReadVariableOp_2)layer0/while/lstm_cell_1/ReadVariableOp_22V
)layer0/while/lstm_cell_1/ReadVariableOp_3)layer0/while/lstm_cell_1/ReadVariableOp_32^
-layer0/while/lstm_cell_1/split/ReadVariableOp-layer0/while/lstm_cell_1/split/ReadVariableOp2b
/layer0/while/lstm_cell_1/split_1/ReadVariableOp/layer0/while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
'__inference_layer0_layer_call_fn_351038

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer0_layer_call_and_return_conditional_losses_3485542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????P:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
??
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_349691

inputs4
0layer0_lstm_cell_1_split_readvariableop_resource6
2layer0_lstm_cell_1_split_1_readvariableop_resource.
*layer0_lstm_cell_1_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?!layer0/lstm_cell_1/ReadVariableOp?#layer0/lstm_cell_1/ReadVariableOp_1?#layer0/lstm_cell_1/ReadVariableOp_2?#layer0/lstm_cell_1/ReadVariableOp_3?'layer0/lstm_cell_1/split/ReadVariableOp?)layer0/lstm_cell_1/split_1/ReadVariableOp?layer0/while?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOpR
layer0/ShapeShapeinputs*
T0*
_output_shapes
:2
layer0/Shape?
layer0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
layer0/strided_slice/stack?
layer0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
layer0/strided_slice/stack_1?
layer0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
layer0/strided_slice/stack_2?
layer0/strided_sliceStridedSlicelayer0/Shape:output:0#layer0/strided_slice/stack:output:0%layer0/strided_slice/stack_1:output:0%layer0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer0/strided_slicek
layer0/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
layer0/zeros/mul/y?
layer0/zeros/mulMullayer0/strided_slice:output:0layer0/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
layer0/zeros/mulm
layer0/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
layer0/zeros/Less/y?
layer0/zeros/LessLesslayer0/zeros/mul:z:0layer0/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
layer0/zeros/Lessq
layer0/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
layer0/zeros/packed/1?
layer0/zeros/packedPacklayer0/strided_slice:output:0layer0/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
layer0/zeros/packedm
layer0/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
layer0/zeros/Const?
layer0/zerosFilllayer0/zeros/packed:output:0layer0/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
layer0/zeroso
layer0/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
layer0/zeros_1/mul/y?
layer0/zeros_1/mulMullayer0/strided_slice:output:0layer0/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
layer0/zeros_1/mulq
layer0/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
layer0/zeros_1/Less/y?
layer0/zeros_1/LessLesslayer0/zeros_1/mul:z:0layer0/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
layer0/zeros_1/Lessu
layer0/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
layer0/zeros_1/packed/1?
layer0/zeros_1/packedPacklayer0/strided_slice:output:0 layer0/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
layer0/zeros_1/packedq
layer0/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
layer0/zeros_1/Const?
layer0/zeros_1Filllayer0/zeros_1/packed:output:0layer0/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
layer0/zeros_1?
layer0/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
layer0/transpose/perm?
layer0/transpose	Transposeinputslayer0/transpose/perm:output:0*
T0*+
_output_shapes
:?????????P2
layer0/transposed
layer0/Shape_1Shapelayer0/transpose:y:0*
T0*
_output_shapes
:2
layer0/Shape_1?
layer0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
layer0/strided_slice_1/stack?
layer0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
layer0/strided_slice_1/stack_1?
layer0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
layer0/strided_slice_1/stack_2?
layer0/strided_slice_1StridedSlicelayer0/Shape_1:output:0%layer0/strided_slice_1/stack:output:0'layer0/strided_slice_1/stack_1:output:0'layer0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer0/strided_slice_1?
"layer0/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"layer0/TensorArrayV2/element_shape?
layer0/TensorArrayV2TensorListReserve+layer0/TensorArrayV2/element_shape:output:0layer0/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
layer0/TensorArrayV2?
<layer0/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   2>
<layer0/TensorArrayUnstack/TensorListFromTensor/element_shape?
.layer0/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlayer0/transpose:y:0Elayer0/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.layer0/TensorArrayUnstack/TensorListFromTensor?
layer0/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
layer0/strided_slice_2/stack?
layer0/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
layer0/strided_slice_2/stack_1?
layer0/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
layer0/strided_slice_2/stack_2?
layer0/strided_slice_2StridedSlicelayer0/transpose:y:0%layer0/strided_slice_2/stack:output:0'layer0/strided_slice_2/stack_1:output:0'layer0/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask2
layer0/strided_slice_2?
"layer0/lstm_cell_1/ones_like/ShapeShapelayer0/strided_slice_2:output:0*
T0*
_output_shapes
:2$
"layer0/lstm_cell_1/ones_like/Shape?
"layer0/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"layer0/lstm_cell_1/ones_like/Const?
layer0/lstm_cell_1/ones_likeFill+layer0/lstm_cell_1/ones_like/Shape:output:0+layer0/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????P2
layer0/lstm_cell_1/ones_like?
$layer0/lstm_cell_1/ones_like_1/ShapeShapelayer0/zeros:output:0*
T0*
_output_shapes
:2&
$layer0/lstm_cell_1/ones_like_1/Shape?
$layer0/lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$layer0/lstm_cell_1/ones_like_1/Const?
layer0/lstm_cell_1/ones_like_1Fill-layer0/lstm_cell_1/ones_like_1/Shape:output:0-layer0/lstm_cell_1/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2 
layer0/lstm_cell_1/ones_like_1?
layer0/lstm_cell_1/mulMullayer0/strided_slice_2:output:0%layer0/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
layer0/lstm_cell_1/mul?
layer0/lstm_cell_1/mul_1Mullayer0/strided_slice_2:output:0%layer0/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
layer0/lstm_cell_1/mul_1?
layer0/lstm_cell_1/mul_2Mullayer0/strided_slice_2:output:0%layer0/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
layer0/lstm_cell_1/mul_2?
layer0/lstm_cell_1/mul_3Mullayer0/strided_slice_2:output:0%layer0/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
layer0/lstm_cell_1/mul_3v
layer0/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
layer0/lstm_cell_1/Const?
"layer0/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"layer0/lstm_cell_1/split/split_dim?
'layer0/lstm_cell_1/split/ReadVariableOpReadVariableOp0layer0_lstm_cell_1_split_readvariableop_resource*
_output_shapes
:	P? *
dtype02)
'layer0/lstm_cell_1/split/ReadVariableOp?
layer0/lstm_cell_1/splitSplit+layer0/lstm_cell_1/split/split_dim:output:0/layer0/lstm_cell_1/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	P?:	P?:	P?:	P?*
	num_split2
layer0/lstm_cell_1/split?
layer0/lstm_cell_1/MatMulMatMullayer0/lstm_cell_1/mul:z:0!layer0/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/MatMul?
layer0/lstm_cell_1/MatMul_1MatMullayer0/lstm_cell_1/mul_1:z:0!layer0/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/MatMul_1?
layer0/lstm_cell_1/MatMul_2MatMullayer0/lstm_cell_1/mul_2:z:0!layer0/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/MatMul_2?
layer0/lstm_cell_1/MatMul_3MatMullayer0/lstm_cell_1/mul_3:z:0!layer0/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/MatMul_3z
layer0/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
layer0/lstm_cell_1/Const_1?
$layer0/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$layer0/lstm_cell_1/split_1/split_dim?
)layer0/lstm_cell_1/split_1/ReadVariableOpReadVariableOp2layer0_lstm_cell_1_split_1_readvariableop_resource*
_output_shapes	
:? *
dtype02+
)layer0/lstm_cell_1/split_1/ReadVariableOp?
layer0/lstm_cell_1/split_1Split-layer0/lstm_cell_1/split_1/split_dim:output:01layer0/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
layer0/lstm_cell_1/split_1?
layer0/lstm_cell_1/BiasAddBiasAdd#layer0/lstm_cell_1/MatMul:product:0#layer0/lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/BiasAdd?
layer0/lstm_cell_1/BiasAdd_1BiasAdd%layer0/lstm_cell_1/MatMul_1:product:0#layer0/lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/BiasAdd_1?
layer0/lstm_cell_1/BiasAdd_2BiasAdd%layer0/lstm_cell_1/MatMul_2:product:0#layer0/lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/BiasAdd_2?
layer0/lstm_cell_1/BiasAdd_3BiasAdd%layer0/lstm_cell_1/MatMul_3:product:0#layer0/lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/BiasAdd_3?
layer0/lstm_cell_1/mul_4Mullayer0/zeros:output:0'layer0/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/mul_4?
layer0/lstm_cell_1/mul_5Mullayer0/zeros:output:0'layer0/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/mul_5?
layer0/lstm_cell_1/mul_6Mullayer0/zeros:output:0'layer0/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/mul_6?
layer0/lstm_cell_1/mul_7Mullayer0/zeros:output:0'layer0/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/mul_7?
!layer0/lstm_cell_1/ReadVariableOpReadVariableOp*layer0_lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02#
!layer0/lstm_cell_1/ReadVariableOp?
&layer0/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&layer0/lstm_cell_1/strided_slice/stack?
(layer0/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2*
(layer0/lstm_cell_1/strided_slice/stack_1?
(layer0/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(layer0/lstm_cell_1/strided_slice/stack_2?
 layer0/lstm_cell_1/strided_sliceStridedSlice)layer0/lstm_cell_1/ReadVariableOp:value:0/layer0/lstm_cell_1/strided_slice/stack:output:01layer0/lstm_cell_1/strided_slice/stack_1:output:01layer0/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 layer0/lstm_cell_1/strided_slice?
layer0/lstm_cell_1/MatMul_4MatMullayer0/lstm_cell_1/mul_4:z:0)layer0/lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/MatMul_4?
layer0/lstm_cell_1/addAddV2#layer0/lstm_cell_1/BiasAdd:output:0%layer0/lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/add?
layer0/lstm_cell_1/SigmoidSigmoidlayer0/lstm_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/Sigmoid?
#layer0/lstm_cell_1/ReadVariableOp_1ReadVariableOp*layer0_lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02%
#layer0/lstm_cell_1/ReadVariableOp_1?
(layer0/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2*
(layer0/lstm_cell_1/strided_slice_1/stack?
*layer0/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2,
*layer0/lstm_cell_1/strided_slice_1/stack_1?
*layer0/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*layer0/lstm_cell_1/strided_slice_1/stack_2?
"layer0/lstm_cell_1/strided_slice_1StridedSlice+layer0/lstm_cell_1/ReadVariableOp_1:value:01layer0/lstm_cell_1/strided_slice_1/stack:output:03layer0/lstm_cell_1/strided_slice_1/stack_1:output:03layer0/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2$
"layer0/lstm_cell_1/strided_slice_1?
layer0/lstm_cell_1/MatMul_5MatMullayer0/lstm_cell_1/mul_5:z:0+layer0/lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/MatMul_5?
layer0/lstm_cell_1/add_1AddV2%layer0/lstm_cell_1/BiasAdd_1:output:0%layer0/lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/add_1?
layer0/lstm_cell_1/Sigmoid_1Sigmoidlayer0/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/Sigmoid_1?
layer0/lstm_cell_1/mul_8Mul layer0/lstm_cell_1/Sigmoid_1:y:0layer0/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/mul_8?
#layer0/lstm_cell_1/ReadVariableOp_2ReadVariableOp*layer0_lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02%
#layer0/lstm_cell_1/ReadVariableOp_2?
(layer0/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2*
(layer0/lstm_cell_1/strided_slice_2/stack?
*layer0/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2,
*layer0/lstm_cell_1/strided_slice_2/stack_1?
*layer0/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*layer0/lstm_cell_1/strided_slice_2/stack_2?
"layer0/lstm_cell_1/strided_slice_2StridedSlice+layer0/lstm_cell_1/ReadVariableOp_2:value:01layer0/lstm_cell_1/strided_slice_2/stack:output:03layer0/lstm_cell_1/strided_slice_2/stack_1:output:03layer0/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2$
"layer0/lstm_cell_1/strided_slice_2?
layer0/lstm_cell_1/MatMul_6MatMullayer0/lstm_cell_1/mul_6:z:0+layer0/lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/MatMul_6?
layer0/lstm_cell_1/add_2AddV2%layer0/lstm_cell_1/BiasAdd_2:output:0%layer0/lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/add_2?
layer0/lstm_cell_1/TanhTanhlayer0/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/Tanh?
layer0/lstm_cell_1/mul_9Mullayer0/lstm_cell_1/Sigmoid:y:0layer0/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/mul_9?
layer0/lstm_cell_1/add_3AddV2layer0/lstm_cell_1/mul_8:z:0layer0/lstm_cell_1/mul_9:z:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/add_3?
#layer0/lstm_cell_1/ReadVariableOp_3ReadVariableOp*layer0_lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02%
#layer0/lstm_cell_1/ReadVariableOp_3?
(layer0/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2*
(layer0/lstm_cell_1/strided_slice_3/stack?
*layer0/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*layer0/lstm_cell_1/strided_slice_3/stack_1?
*layer0/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*layer0/lstm_cell_1/strided_slice_3/stack_2?
"layer0/lstm_cell_1/strided_slice_3StridedSlice+layer0/lstm_cell_1/ReadVariableOp_3:value:01layer0/lstm_cell_1/strided_slice_3/stack:output:03layer0/lstm_cell_1/strided_slice_3/stack_1:output:03layer0/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2$
"layer0/lstm_cell_1/strided_slice_3?
layer0/lstm_cell_1/MatMul_7MatMullayer0/lstm_cell_1/mul_7:z:0+layer0/lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/MatMul_7?
layer0/lstm_cell_1/add_4AddV2%layer0/lstm_cell_1/BiasAdd_3:output:0%layer0/lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/add_4?
layer0/lstm_cell_1/Sigmoid_2Sigmoidlayer0/lstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/Sigmoid_2?
layer0/lstm_cell_1/Tanh_1Tanhlayer0/lstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/Tanh_1?
layer0/lstm_cell_1/mul_10Mul layer0/lstm_cell_1/Sigmoid_2:y:0layer0/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/mul_10?
$layer0/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2&
$layer0/TensorArrayV2_1/element_shape?
layer0/TensorArrayV2_1TensorListReserve-layer0/TensorArrayV2_1/element_shape:output:0layer0/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
layer0/TensorArrayV2_1\
layer0/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
layer0/time?
layer0/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
layer0/while/maximum_iterationsx
layer0/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
layer0/while/loop_counter?
layer0/whileWhile"layer0/while/loop_counter:output:0(layer0/while/maximum_iterations:output:0layer0/time:output:0layer0/TensorArrayV2_1:handle:0layer0/zeros:output:0layer0/zeros_1:output:0layer0/strided_slice_1:output:0>layer0/TensorArrayUnstack/TensorListFromTensor:output_handle:00layer0_lstm_cell_1_split_readvariableop_resource2layer0_lstm_cell_1_split_1_readvariableop_resource*layer0_lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*$
bodyR
layer0_while_body_349541*$
condR
layer0_while_cond_349540*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
layer0/while?
7layer0/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7layer0/TensorArrayV2Stack/TensorListStack/element_shape?
)layer0/TensorArrayV2Stack/TensorListStackTensorListStacklayer0/while:output:3@layer0/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02+
)layer0/TensorArrayV2Stack/TensorListStack?
layer0/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
layer0/strided_slice_3/stack?
layer0/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
layer0/strided_slice_3/stack_1?
layer0/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
layer0/strided_slice_3/stack_2?
layer0/strided_slice_3StridedSlice2layer0/TensorArrayV2Stack/TensorListStack:tensor:0%layer0/strided_slice_3/stack:output:0'layer0/strided_slice_3/stack_1:output:0'layer0/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
layer0/strided_slice_3?
layer0/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
layer0/transpose_1/perm?
layer0/transpose_1	Transpose2layer0/TensorArrayV2Stack/TensorListStack:tensor:0 layer0/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
layer0/transpose_1t
layer0/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
layer0/runtime?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_10/MatMul/ReadVariableOp?
dense_10/MatMulMatMullayer0/strided_slice_3:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_10/MatMul?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_10/BiasAddt
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_10/Relu?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMuldense_10/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
output/Softmax?
IdentityIdentityoutput/Softmax:softmax:0 ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp"^layer0/lstm_cell_1/ReadVariableOp$^layer0/lstm_cell_1/ReadVariableOp_1$^layer0/lstm_cell_1/ReadVariableOp_2$^layer0/lstm_cell_1/ReadVariableOp_3(^layer0/lstm_cell_1/split/ReadVariableOp*^layer0/lstm_cell_1/split_1/ReadVariableOp^layer0/while^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????P:::::::2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2F
!layer0/lstm_cell_1/ReadVariableOp!layer0/lstm_cell_1/ReadVariableOp2J
#layer0/lstm_cell_1/ReadVariableOp_1#layer0/lstm_cell_1/ReadVariableOp_12J
#layer0/lstm_cell_1/ReadVariableOp_2#layer0/lstm_cell_1/ReadVariableOp_22J
#layer0/lstm_cell_1/ReadVariableOp_3#layer0/lstm_cell_1/ReadVariableOp_32R
'layer0/lstm_cell_1/split/ReadVariableOp'layer0/lstm_cell_1/split/ReadVariableOp2V
)layer0/lstm_cell_1/split_1/ReadVariableOp)layer0/lstm_cell_1/split_1/ReadVariableOp2
layer0/whilelayer0/while2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
??
?

layer0_while_body_349541*
&layer0_while_layer0_while_loop_counter0
,layer0_while_layer0_while_maximum_iterations
layer0_while_placeholder
layer0_while_placeholder_1
layer0_while_placeholder_2
layer0_while_placeholder_3)
%layer0_while_layer0_strided_slice_1_0e
alayer0_while_tensorarrayv2read_tensorlistgetitem_layer0_tensorarrayunstack_tensorlistfromtensor_0<
8layer0_while_lstm_cell_1_split_readvariableop_resource_0>
:layer0_while_lstm_cell_1_split_1_readvariableop_resource_06
2layer0_while_lstm_cell_1_readvariableop_resource_0
layer0_while_identity
layer0_while_identity_1
layer0_while_identity_2
layer0_while_identity_3
layer0_while_identity_4
layer0_while_identity_5'
#layer0_while_layer0_strided_slice_1c
_layer0_while_tensorarrayv2read_tensorlistgetitem_layer0_tensorarrayunstack_tensorlistfromtensor:
6layer0_while_lstm_cell_1_split_readvariableop_resource<
8layer0_while_lstm_cell_1_split_1_readvariableop_resource4
0layer0_while_lstm_cell_1_readvariableop_resource??'layer0/while/lstm_cell_1/ReadVariableOp?)layer0/while/lstm_cell_1/ReadVariableOp_1?)layer0/while/lstm_cell_1/ReadVariableOp_2?)layer0/while/lstm_cell_1/ReadVariableOp_3?-layer0/while/lstm_cell_1/split/ReadVariableOp?/layer0/while/lstm_cell_1/split_1/ReadVariableOp?
>layer0/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   2@
>layer0/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0layer0/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalayer0_while_tensorarrayv2read_tensorlistgetitem_layer0_tensorarrayunstack_tensorlistfromtensor_0layer0_while_placeholderGlayer0/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype022
0layer0/while/TensorArrayV2Read/TensorListGetItem?
(layer0/while/lstm_cell_1/ones_like/ShapeShape7layer0/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2*
(layer0/while/lstm_cell_1/ones_like/Shape?
(layer0/while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(layer0/while/lstm_cell_1/ones_like/Const?
"layer0/while/lstm_cell_1/ones_likeFill1layer0/while/lstm_cell_1/ones_like/Shape:output:01layer0/while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????P2$
"layer0/while/lstm_cell_1/ones_like?
*layer0/while/lstm_cell_1/ones_like_1/ShapeShapelayer0_while_placeholder_2*
T0*
_output_shapes
:2,
*layer0/while/lstm_cell_1/ones_like_1/Shape?
*layer0/while/lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*layer0/while/lstm_cell_1/ones_like_1/Const?
$layer0/while/lstm_cell_1/ones_like_1Fill3layer0/while/lstm_cell_1/ones_like_1/Shape:output:03layer0/while/lstm_cell_1/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2&
$layer0/while/lstm_cell_1/ones_like_1?
layer0/while/lstm_cell_1/mulMul7layer0/while/TensorArrayV2Read/TensorListGetItem:item:0+layer0/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
layer0/while/lstm_cell_1/mul?
layer0/while/lstm_cell_1/mul_1Mul7layer0/while/TensorArrayV2Read/TensorListGetItem:item:0+layer0/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2 
layer0/while/lstm_cell_1/mul_1?
layer0/while/lstm_cell_1/mul_2Mul7layer0/while/TensorArrayV2Read/TensorListGetItem:item:0+layer0/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2 
layer0/while/lstm_cell_1/mul_2?
layer0/while/lstm_cell_1/mul_3Mul7layer0/while/TensorArrayV2Read/TensorListGetItem:item:0+layer0/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2 
layer0/while/lstm_cell_1/mul_3?
layer0/while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
layer0/while/lstm_cell_1/Const?
(layer0/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(layer0/while/lstm_cell_1/split/split_dim?
-layer0/while/lstm_cell_1/split/ReadVariableOpReadVariableOp8layer0_while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes
:	P? *
dtype02/
-layer0/while/lstm_cell_1/split/ReadVariableOp?
layer0/while/lstm_cell_1/splitSplit1layer0/while/lstm_cell_1/split/split_dim:output:05layer0/while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	P?:	P?:	P?:	P?*
	num_split2 
layer0/while/lstm_cell_1/split?
layer0/while/lstm_cell_1/MatMulMatMul layer0/while/lstm_cell_1/mul:z:0'layer0/while/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????2!
layer0/while/lstm_cell_1/MatMul?
!layer0/while/lstm_cell_1/MatMul_1MatMul"layer0/while/lstm_cell_1/mul_1:z:0'layer0/while/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????2#
!layer0/while/lstm_cell_1/MatMul_1?
!layer0/while/lstm_cell_1/MatMul_2MatMul"layer0/while/lstm_cell_1/mul_2:z:0'layer0/while/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????2#
!layer0/while/lstm_cell_1/MatMul_2?
!layer0/while/lstm_cell_1/MatMul_3MatMul"layer0/while/lstm_cell_1/mul_3:z:0'layer0/while/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????2#
!layer0/while/lstm_cell_1/MatMul_3?
 layer0/while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2"
 layer0/while/lstm_cell_1/Const_1?
*layer0/while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*layer0/while/lstm_cell_1/split_1/split_dim?
/layer0/while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp:layer0_while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes	
:? *
dtype021
/layer0/while/lstm_cell_1/split_1/ReadVariableOp?
 layer0/while/lstm_cell_1/split_1Split3layer0/while/lstm_cell_1/split_1/split_dim:output:07layer0/while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2"
 layer0/while/lstm_cell_1/split_1?
 layer0/while/lstm_cell_1/BiasAddBiasAdd)layer0/while/lstm_cell_1/MatMul:product:0)layer0/while/lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:??????????2"
 layer0/while/lstm_cell_1/BiasAdd?
"layer0/while/lstm_cell_1/BiasAdd_1BiasAdd+layer0/while/lstm_cell_1/MatMul_1:product:0)layer0/while/lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:??????????2$
"layer0/while/lstm_cell_1/BiasAdd_1?
"layer0/while/lstm_cell_1/BiasAdd_2BiasAdd+layer0/while/lstm_cell_1/MatMul_2:product:0)layer0/while/lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:??????????2$
"layer0/while/lstm_cell_1/BiasAdd_2?
"layer0/while/lstm_cell_1/BiasAdd_3BiasAdd+layer0/while/lstm_cell_1/MatMul_3:product:0)layer0/while/lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:??????????2$
"layer0/while/lstm_cell_1/BiasAdd_3?
layer0/while/lstm_cell_1/mul_4Mullayer0_while_placeholder_2-layer0/while/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2 
layer0/while/lstm_cell_1/mul_4?
layer0/while/lstm_cell_1/mul_5Mullayer0_while_placeholder_2-layer0/while/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2 
layer0/while/lstm_cell_1/mul_5?
layer0/while/lstm_cell_1/mul_6Mullayer0_while_placeholder_2-layer0/while/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2 
layer0/while/lstm_cell_1/mul_6?
layer0/while/lstm_cell_1/mul_7Mullayer0_while_placeholder_2-layer0/while/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2 
layer0/while/lstm_cell_1/mul_7?
'layer0/while/lstm_cell_1/ReadVariableOpReadVariableOp2layer0_while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02)
'layer0/while/lstm_cell_1/ReadVariableOp?
,layer0/while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,layer0/while/lstm_cell_1/strided_slice/stack?
.layer0/while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.layer0/while/lstm_cell_1/strided_slice/stack_1?
.layer0/while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.layer0/while/lstm_cell_1/strided_slice/stack_2?
&layer0/while/lstm_cell_1/strided_sliceStridedSlice/layer0/while/lstm_cell_1/ReadVariableOp:value:05layer0/while/lstm_cell_1/strided_slice/stack:output:07layer0/while/lstm_cell_1/strided_slice/stack_1:output:07layer0/while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&layer0/while/lstm_cell_1/strided_slice?
!layer0/while/lstm_cell_1/MatMul_4MatMul"layer0/while/lstm_cell_1/mul_4:z:0/layer0/while/lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2#
!layer0/while/lstm_cell_1/MatMul_4?
layer0/while/lstm_cell_1/addAddV2)layer0/while/lstm_cell_1/BiasAdd:output:0+layer0/while/lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
layer0/while/lstm_cell_1/add?
 layer0/while/lstm_cell_1/SigmoidSigmoid layer0/while/lstm_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2"
 layer0/while/lstm_cell_1/Sigmoid?
)layer0/while/lstm_cell_1/ReadVariableOp_1ReadVariableOp2layer0_while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02+
)layer0/while/lstm_cell_1/ReadVariableOp_1?
.layer0/while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.layer0/while/lstm_cell_1/strided_slice_1/stack?
0layer0/while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0layer0/while/lstm_cell_1/strided_slice_1/stack_1?
0layer0/while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0layer0/while/lstm_cell_1/strided_slice_1/stack_2?
(layer0/while/lstm_cell_1/strided_slice_1StridedSlice1layer0/while/lstm_cell_1/ReadVariableOp_1:value:07layer0/while/lstm_cell_1/strided_slice_1/stack:output:09layer0/while/lstm_cell_1/strided_slice_1/stack_1:output:09layer0/while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2*
(layer0/while/lstm_cell_1/strided_slice_1?
!layer0/while/lstm_cell_1/MatMul_5MatMul"layer0/while/lstm_cell_1/mul_5:z:01layer0/while/lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2#
!layer0/while/lstm_cell_1/MatMul_5?
layer0/while/lstm_cell_1/add_1AddV2+layer0/while/lstm_cell_1/BiasAdd_1:output:0+layer0/while/lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2 
layer0/while/lstm_cell_1/add_1?
"layer0/while/lstm_cell_1/Sigmoid_1Sigmoid"layer0/while/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2$
"layer0/while/lstm_cell_1/Sigmoid_1?
layer0/while/lstm_cell_1/mul_8Mul&layer0/while/lstm_cell_1/Sigmoid_1:y:0layer0_while_placeholder_3*
T0*(
_output_shapes
:??????????2 
layer0/while/lstm_cell_1/mul_8?
)layer0/while/lstm_cell_1/ReadVariableOp_2ReadVariableOp2layer0_while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02+
)layer0/while/lstm_cell_1/ReadVariableOp_2?
.layer0/while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.layer0/while/lstm_cell_1/strided_slice_2/stack?
0layer0/while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0layer0/while/lstm_cell_1/strided_slice_2/stack_1?
0layer0/while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0layer0/while/lstm_cell_1/strided_slice_2/stack_2?
(layer0/while/lstm_cell_1/strided_slice_2StridedSlice1layer0/while/lstm_cell_1/ReadVariableOp_2:value:07layer0/while/lstm_cell_1/strided_slice_2/stack:output:09layer0/while/lstm_cell_1/strided_slice_2/stack_1:output:09layer0/while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2*
(layer0/while/lstm_cell_1/strided_slice_2?
!layer0/while/lstm_cell_1/MatMul_6MatMul"layer0/while/lstm_cell_1/mul_6:z:01layer0/while/lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2#
!layer0/while/lstm_cell_1/MatMul_6?
layer0/while/lstm_cell_1/add_2AddV2+layer0/while/lstm_cell_1/BiasAdd_2:output:0+layer0/while/lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2 
layer0/while/lstm_cell_1/add_2?
layer0/while/lstm_cell_1/TanhTanh"layer0/while/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
layer0/while/lstm_cell_1/Tanh?
layer0/while/lstm_cell_1/mul_9Mul$layer0/while/lstm_cell_1/Sigmoid:y:0!layer0/while/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2 
layer0/while/lstm_cell_1/mul_9?
layer0/while/lstm_cell_1/add_3AddV2"layer0/while/lstm_cell_1/mul_8:z:0"layer0/while/lstm_cell_1/mul_9:z:0*
T0*(
_output_shapes
:??????????2 
layer0/while/lstm_cell_1/add_3?
)layer0/while/lstm_cell_1/ReadVariableOp_3ReadVariableOp2layer0_while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02+
)layer0/while/lstm_cell_1/ReadVariableOp_3?
.layer0/while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.layer0/while/lstm_cell_1/strided_slice_3/stack?
0layer0/while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0layer0/while/lstm_cell_1/strided_slice_3/stack_1?
0layer0/while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0layer0/while/lstm_cell_1/strided_slice_3/stack_2?
(layer0/while/lstm_cell_1/strided_slice_3StridedSlice1layer0/while/lstm_cell_1/ReadVariableOp_3:value:07layer0/while/lstm_cell_1/strided_slice_3/stack:output:09layer0/while/lstm_cell_1/strided_slice_3/stack_1:output:09layer0/while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2*
(layer0/while/lstm_cell_1/strided_slice_3?
!layer0/while/lstm_cell_1/MatMul_7MatMul"layer0/while/lstm_cell_1/mul_7:z:01layer0/while/lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2#
!layer0/while/lstm_cell_1/MatMul_7?
layer0/while/lstm_cell_1/add_4AddV2+layer0/while/lstm_cell_1/BiasAdd_3:output:0+layer0/while/lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2 
layer0/while/lstm_cell_1/add_4?
"layer0/while/lstm_cell_1/Sigmoid_2Sigmoid"layer0/while/lstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:??????????2$
"layer0/while/lstm_cell_1/Sigmoid_2?
layer0/while/lstm_cell_1/Tanh_1Tanh"layer0/while/lstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:??????????2!
layer0/while/lstm_cell_1/Tanh_1?
layer0/while/lstm_cell_1/mul_10Mul&layer0/while/lstm_cell_1/Sigmoid_2:y:0#layer0/while/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2!
layer0/while/lstm_cell_1/mul_10?
1layer0/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlayer0_while_placeholder_1layer0_while_placeholder#layer0/while/lstm_cell_1/mul_10:z:0*
_output_shapes
: *
element_dtype023
1layer0/while/TensorArrayV2Write/TensorListSetItemj
layer0/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
layer0/while/add/y?
layer0/while/addAddV2layer0_while_placeholderlayer0/while/add/y:output:0*
T0*
_output_shapes
: 2
layer0/while/addn
layer0/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
layer0/while/add_1/y?
layer0/while/add_1AddV2&layer0_while_layer0_while_loop_counterlayer0/while/add_1/y:output:0*
T0*
_output_shapes
: 2
layer0/while/add_1?
layer0/while/IdentityIdentitylayer0/while/add_1:z:0(^layer0/while/lstm_cell_1/ReadVariableOp*^layer0/while/lstm_cell_1/ReadVariableOp_1*^layer0/while/lstm_cell_1/ReadVariableOp_2*^layer0/while/lstm_cell_1/ReadVariableOp_3.^layer0/while/lstm_cell_1/split/ReadVariableOp0^layer0/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
layer0/while/Identity?
layer0/while/Identity_1Identity,layer0_while_layer0_while_maximum_iterations(^layer0/while/lstm_cell_1/ReadVariableOp*^layer0/while/lstm_cell_1/ReadVariableOp_1*^layer0/while/lstm_cell_1/ReadVariableOp_2*^layer0/while/lstm_cell_1/ReadVariableOp_3.^layer0/while/lstm_cell_1/split/ReadVariableOp0^layer0/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
layer0/while/Identity_1?
layer0/while/Identity_2Identitylayer0/while/add:z:0(^layer0/while/lstm_cell_1/ReadVariableOp*^layer0/while/lstm_cell_1/ReadVariableOp_1*^layer0/while/lstm_cell_1/ReadVariableOp_2*^layer0/while/lstm_cell_1/ReadVariableOp_3.^layer0/while/lstm_cell_1/split/ReadVariableOp0^layer0/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
layer0/while/Identity_2?
layer0/while/Identity_3IdentityAlayer0/while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^layer0/while/lstm_cell_1/ReadVariableOp*^layer0/while/lstm_cell_1/ReadVariableOp_1*^layer0/while/lstm_cell_1/ReadVariableOp_2*^layer0/while/lstm_cell_1/ReadVariableOp_3.^layer0/while/lstm_cell_1/split/ReadVariableOp0^layer0/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
layer0/while/Identity_3?
layer0/while/Identity_4Identity#layer0/while/lstm_cell_1/mul_10:z:0(^layer0/while/lstm_cell_1/ReadVariableOp*^layer0/while/lstm_cell_1/ReadVariableOp_1*^layer0/while/lstm_cell_1/ReadVariableOp_2*^layer0/while/lstm_cell_1/ReadVariableOp_3.^layer0/while/lstm_cell_1/split/ReadVariableOp0^layer0/while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
layer0/while/Identity_4?
layer0/while/Identity_5Identity"layer0/while/lstm_cell_1/add_3:z:0(^layer0/while/lstm_cell_1/ReadVariableOp*^layer0/while/lstm_cell_1/ReadVariableOp_1*^layer0/while/lstm_cell_1/ReadVariableOp_2*^layer0/while/lstm_cell_1/ReadVariableOp_3.^layer0/while/lstm_cell_1/split/ReadVariableOp0^layer0/while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
layer0/while/Identity_5"7
layer0_while_identitylayer0/while/Identity:output:0";
layer0_while_identity_1 layer0/while/Identity_1:output:0";
layer0_while_identity_2 layer0/while/Identity_2:output:0";
layer0_while_identity_3 layer0/while/Identity_3:output:0";
layer0_while_identity_4 layer0/while/Identity_4:output:0";
layer0_while_identity_5 layer0/while/Identity_5:output:0"L
#layer0_while_layer0_strided_slice_1%layer0_while_layer0_strided_slice_1_0"f
0layer0_while_lstm_cell_1_readvariableop_resource2layer0_while_lstm_cell_1_readvariableop_resource_0"v
8layer0_while_lstm_cell_1_split_1_readvariableop_resource:layer0_while_lstm_cell_1_split_1_readvariableop_resource_0"r
6layer0_while_lstm_cell_1_split_readvariableop_resource8layer0_while_lstm_cell_1_split_readvariableop_resource_0"?
_layer0_while_tensorarrayv2read_tensorlistgetitem_layer0_tensorarrayunstack_tensorlistfromtensoralayer0_while_tensorarrayv2read_tensorlistgetitem_layer0_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2R
'layer0/while/lstm_cell_1/ReadVariableOp'layer0/while/lstm_cell_1/ReadVariableOp2V
)layer0/while/lstm_cell_1/ReadVariableOp_1)layer0/while/lstm_cell_1/ReadVariableOp_12V
)layer0/while/lstm_cell_1/ReadVariableOp_2)layer0/while/lstm_cell_1/ReadVariableOp_22V
)layer0/while/lstm_cell_1/ReadVariableOp_3)layer0/while/lstm_cell_1/ReadVariableOp_32^
-layer0/while/lstm_cell_1/split/ReadVariableOp-layer0/while/lstm_cell_1/split/ReadVariableOp2b
/layer0/while/lstm_cell_1/split_1/ReadVariableOp/layer0/while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?D
?
B__inference_layer0_layer_call_and_return_conditional_losses_348159

inputs
lstm_cell_1_348077
lstm_cell_1_348079
lstm_cell_1_348081
identity??#lstm_cell_1/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????P2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask2
strided_slice_2?
#lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1_348077lstm_cell_1_348079lstm_cell_1_348081*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_3476642%
#lstm_cell_1/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1_348077lstm_cell_1_348079lstm_cell_1_348081*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_348090*
condR
while_cond_348089*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_1/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????P:::2J
#lstm_cell_1/StatefulPartitionedCall#lstm_cell_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????P
 
_user_specified_nameinputs
?C
?
__inference__traced_save_351468
file_prefix.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_layer0_lstm_cell_1_kernel_read_readvariableopB
>savev2_layer0_lstm_cell_1_recurrent_kernel_read_readvariableop6
2savev2_layer0_lstm_cell_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_10_kernel_m_read_readvariableop3
/savev2_adam_dense_10_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop?
;savev2_adam_layer0_lstm_cell_1_kernel_m_read_readvariableopI
Esavev2_adam_layer0_lstm_cell_1_recurrent_kernel_m_read_readvariableop=
9savev2_adam_layer0_lstm_cell_1_bias_m_read_readvariableop5
1savev2_adam_dense_10_kernel_v_read_readvariableop3
/savev2_adam_dense_10_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop?
;savev2_adam_layer0_lstm_cell_1_kernel_v_read_readvariableopI
Esavev2_adam_layer0_lstm_cell_1_recurrent_kernel_v_read_readvariableop=
9savev2_adam_layer0_lstm_cell_1_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_layer0_lstm_cell_1_kernel_read_readvariableop>savev2_layer0_lstm_cell_1_recurrent_kernel_read_readvariableop2savev2_layer0_lstm_cell_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_10_kernel_m_read_readvariableop/savev2_adam_dense_10_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop;savev2_adam_layer0_lstm_cell_1_kernel_m_read_readvariableopEsavev2_adam_layer0_lstm_cell_1_recurrent_kernel_m_read_readvariableop9savev2_adam_layer0_lstm_cell_1_bias_m_read_readvariableop1savev2_adam_dense_10_kernel_v_read_readvariableop/savev2_adam_dense_10_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableop;savev2_adam_layer0_lstm_cell_1_kernel_v_read_readvariableopEsavev2_adam_layer0_lstm_cell_1_recurrent_kernel_v_read_readvariableop9savev2_adam_layer0_lstm_cell_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *-
dtypes#
!2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:	?:: : : : : :	P? :
?? :? : : : : :
??:?:	?::	P? :
?? :? :
??:?:	?::	P? :
?? :? : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :%
!

_output_shapes
:	P? :&"
 
_output_shapes
:
?? :!

_output_shapes	
:? :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	P? :&"
 
_output_shapes
:
?? :!

_output_shapes	
:? :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	P? :&"
 
_output_shapes
:
?? :!

_output_shapes	
:? :

_output_shapes
: 
?N
?
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_351321

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpX
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????P2
	ones_like^
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
ones_like_1_
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:?????????P2
mulc
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????P2
mul_1c
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????P2
mul_2c
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????P2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	P? *
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	P?:	P?:	P?:	P?*
	num_split2
splitf
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:??????????2
MatMull
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:??????????2

MatMul_1l
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:??????????2

MatMul_2l
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:??????????2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:? *
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????2
	BiasAdd_3h
mul_4Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
mul_4h
mul_5Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
mul_5h
mul_6Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
mul_6h
mul_7Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
mul_7z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
?? *
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slicet
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
?? *
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_1v
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1a
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mul_8~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
?? *
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_2v
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????2
Tanh_
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????2
mul_9`
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:??????????2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
?? *
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_3v
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:??????????2
Tanh_1e
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
mul_10?
IdentityIdentity
mul_10:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity
mul_10:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????P:??????????:??????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_348915
layer0_input
layer0_348897
layer0_348899
layer0_348901
dense_10_348904
dense_10_348906
output_348909
output_348911
identity?? dense_10/StatefulPartitionedCall?layer0/StatefulPartitionedCall?output/StatefulPartitionedCall?
layer0/StatefulPartitionedCallStatefulPartitionedCalllayer0_inputlayer0_348897layer0_348899layer0_348901*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer0_layer_call_and_return_conditional_losses_3488092 
layer0/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall'layer0/StatefulPartitionedCall:output:0dense_10_348904dense_10_348906*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_3488502"
 dense_10/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0output_348909output_348911*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_3488772 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall^layer0/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????P:::::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2@
layer0/StatefulPartitionedCalllayer0/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:Y U
+
_output_shapes
:?????????P
&
_user_specified_namelayer0_input
?
?
while_cond_348353
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_348353___redundant_placeholder04
0while_while_cond_348353___redundant_placeholder14
0while_while_cond_348353___redundant_placeholder24
0while_while_cond_348353___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
$__inference_signature_wrapper_349025
layer0_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_3473922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????P:::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????P
&
_user_specified_namelayer0_input
?%
?
while_body_348090
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_1_348114_0
while_lstm_cell_1_348116_0
while_lstm_cell_1_348118_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_1_348114
while_lstm_cell_1_348116
while_lstm_cell_1_348118??)while/lstm_cell_1/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1_348114_0while_lstm_cell_1_348116_0while_lstm_cell_1_348118_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_3476642+
)while/lstm_cell_1/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/lstm_cell_1/StatefulPartitionedCall:output:1*^while/lstm_cell_1/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity2while/lstm_cell_1/StatefulPartitionedCall:output:2*^while/lstm_cell_1/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_1_348114while_lstm_cell_1_348114_0"6
while_lstm_cell_1_348116while_lstm_cell_1_348116_0"6
while_lstm_cell_1_348118while_lstm_cell_1_348118_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2V
)while/lstm_cell_1/StatefulPartitionedCall)while/lstm_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
'__inference_layer0_layer_call_fn_350378
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer0_layer_call_and_return_conditional_losses_3480272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????P:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????P
"
_user_specified_name
inputs/0
??
?
while_body_350891
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
1while_lstm_cell_1_split_readvariableop_resource_07
3while_lstm_cell_1_split_1_readvariableop_resource_0/
+while_lstm_cell_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
/while_lstm_cell_1_split_readvariableop_resource5
1while_lstm_cell_1_split_1_readvariableop_resource-
)while_lstm_cell_1_readvariableop_resource?? while/lstm_cell_1/ReadVariableOp?"while/lstm_cell_1/ReadVariableOp_1?"while/lstm_cell_1/ReadVariableOp_2?"while/lstm_cell_1/ReadVariableOp_3?&while/lstm_cell_1/split/ReadVariableOp?(while/lstm_cell_1/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
!while/lstm_cell_1/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/ones_like/Shape?
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell_1/ones_like/Const?
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/ones_like?
#while/lstm_cell_1/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_1/ones_like_1/Shape?
#while/lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#while/lstm_cell_1/ones_like_1/Const?
while/lstm_cell_1/ones_like_1Fill,while/lstm_cell_1/ones_like_1/Shape:output:0,while/lstm_cell_1/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/ones_like_1?
while/lstm_cell_1/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/mul?
while/lstm_cell_1/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/mul_1?
while/lstm_cell_1/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/mul_2?
while/lstm_cell_1/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/mul_3t
while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const?
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_1/split/split_dim?
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes
:	P? *
dtype02(
&while/lstm_cell_1/split/ReadVariableOp?
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	P?:	P?:	P?:	P?*
	num_split2
while/lstm_cell_1/split?
while/lstm_cell_1/MatMulMatMulwhile/lstm_cell_1/mul:z:0 while/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul?
while/lstm_cell_1/MatMul_1MatMulwhile/lstm_cell_1/mul_1:z:0 while/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_1?
while/lstm_cell_1/MatMul_2MatMulwhile/lstm_cell_1/mul_2:z:0 while/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_2?
while/lstm_cell_1/MatMul_3MatMulwhile/lstm_cell_1/mul_3:z:0 while/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_3x
while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const_1?
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_1/split_1/split_dim?
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes	
:? *
dtype02*
(while/lstm_cell_1/split_1/ReadVariableOp?
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
while/lstm_cell_1/split_1?
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/BiasAdd?
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/BiasAdd_1?
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/BiasAdd_2?
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/BiasAdd_3?
while/lstm_cell_1/mul_4Mulwhile_placeholder_2&while/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_4?
while/lstm_cell_1/mul_5Mulwhile_placeholder_2&while/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_5?
while/lstm_cell_1/mul_6Mulwhile_placeholder_2&while/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_6?
while/lstm_cell_1/mul_7Mulwhile_placeholder_2&while/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_7?
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02"
 while/lstm_cell_1/ReadVariableOp?
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_1/strided_slice/stack?
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice/stack_1?
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_1/strided_slice/stack_2?
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell_1/strided_slice?
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul_4:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_4?
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add?
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Sigmoid?
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02$
"while/lstm_cell_1/ReadVariableOp_1?
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_1/stack?
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/lstm_cell_1/strided_slice_1/stack_1?
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_1/stack_2?
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_1?
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_5:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_5?
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add_1?
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Sigmoid_1?
while/lstm_cell_1/mul_8Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_8?
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02$
"while/lstm_cell_1/ReadVariableOp_2?
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_2/stack?
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/lstm_cell_1/strided_slice_2/stack_1?
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_2/stack_2?
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_2?
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_6:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_6?
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add_2?
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Tanh?
while/lstm_cell_1/mul_9Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_9?
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_8:z:0while/lstm_cell_1/mul_9:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add_3?
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02$
"while/lstm_cell_1/ReadVariableOp_3?
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_3/stack?
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_1/strided_slice_3/stack_1?
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_3/stack_2?
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_3?
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_7:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_7?
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add_4?
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Sigmoid_2?
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Tanh_1?
while/lstm_cell_1/mul_10Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_10?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_1/mul_10:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_1/add_3:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_347580

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpX
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????P2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:?????????P2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2???2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????P2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_1/Const?
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????P2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2??02(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_1/GreaterEqual/y?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2
dropout_1/GreaterEqual?
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2
dropout_1/Cast?
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????P2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_2/Const?
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????P2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2???2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_2/GreaterEqual/y?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2
dropout_2/GreaterEqual?
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2
dropout_2/Cast?
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????P2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_3/Const?
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????P2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shape?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2???2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_3/GreaterEqual/y?
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2
dropout_3/GreaterEqual?
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2
dropout_3/Cast?
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????P2
dropout_3/Mul_1\
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
ones_like_1g
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_4/Const?
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_4/Mulf
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_4/Shape?
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_4/GreaterEqual/y?
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_4/GreaterEqual?
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_4/Cast?
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_4/Mul_1g
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_5/Const?
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_5/Mulf
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_5/Shape?
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_5/GreaterEqual/y?
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_5/GreaterEqual?
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_5/Cast?
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_5/Mul_1g
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_6/Const?
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_6/Mulf
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_6/Shape?
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2(
&dropout_6/random_uniform/RandomUniformy
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_6/GreaterEqual/y?
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_6/GreaterEqual?
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_6/Cast?
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_6/Mul_1g
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_7/Const?
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_7/Mulf
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_7/Shape?
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2(
&dropout_7/random_uniform/RandomUniformy
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_7/GreaterEqual/y?
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_7/GreaterEqual?
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_7/Cast?
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_7/Mul_1^
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
muld
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
mul_1d
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
mul_2d
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	P? *
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	P?:	P?:	P?:	P?*
	num_split2
splitf
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:??????????2
MatMull
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:??????????2

MatMul_1l
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:??????????2

MatMul_2l
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:??????????2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:? *
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????2
	BiasAdd_3e
mul_4Mulstatesdropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_4e
mul_5Mulstatesdropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_5e
mul_6Mulstatesdropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_6e
mul_7Mulstatesdropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_7z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
?? *
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slicet
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
?? *
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_1v
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1a
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mul_8~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
?? *
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_2v
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????2
Tanh_
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????2
mul_9`
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:??????????2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
?? *
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_3v
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:??????????2
Tanh_1e
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
mul_10?
IdentityIdentity
mul_10:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity
mul_10:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:?????????P:??????????:??????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?	
?
B__inference_output_layer_call_and_return_conditional_losses_348877

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?%
?
while_body_347958
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_1_347982_0
while_lstm_cell_1_347984_0
while_lstm_cell_1_347986_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_1_347982
while_lstm_cell_1_347984
while_lstm_cell_1_347986??)while/lstm_cell_1/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1_347982_0while_lstm_cell_1_347984_0while_lstm_cell_1_347986_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_3475802+
)while/lstm_cell_1/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/lstm_cell_1/StatefulPartitionedCall:output:1*^while/lstm_cell_1/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity2while/lstm_cell_1/StatefulPartitionedCall:output:2*^while/lstm_cell_1/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_1_347982while_lstm_cell_1_347982_0"6
while_lstm_cell_1_347984while_lstm_cell_1_347984_0"6
while_lstm_cell_1_347986while_lstm_cell_1_347986_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2V
)while/lstm_cell_1/StatefulPartitionedCall)while/lstm_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_350890
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_350890___redundant_placeholder04
0while_while_cond_350890___redundant_placeholder14
0while_while_cond_350890___redundant_placeholder24
0while_while_cond_350890___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
~
)__inference_dense_10_layer_call_fn_351069

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_3488502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_348939

inputs
layer0_348921
layer0_348923
layer0_348925
dense_10_348928
dense_10_348930
output_348933
output_348935
identity?? dense_10/StatefulPartitionedCall?layer0/StatefulPartitionedCall?output/StatefulPartitionedCall?
layer0/StatefulPartitionedCallStatefulPartitionedCallinputslayer0_348921layer0_348923layer0_348925*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer0_layer_call_and_return_conditional_losses_3485542 
layer0/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall'layer0/StatefulPartitionedCall:output:0dense_10_348928dense_10_348930*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_3488502"
 dense_10/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0output_348933output_348935*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_3488772 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall^layer0/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????P:::::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2@
layer0/StatefulPartitionedCalllayer0/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
.__inference_sequential_12_layer_call_fn_348996
layer0_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_3489792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????P:::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????P
&
_user_specified_namelayer0_input
??
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_349422

inputs4
0layer0_lstm_cell_1_split_readvariableop_resource6
2layer0_lstm_cell_1_split_1_readvariableop_resource.
*layer0_lstm_cell_1_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?!layer0/lstm_cell_1/ReadVariableOp?#layer0/lstm_cell_1/ReadVariableOp_1?#layer0/lstm_cell_1/ReadVariableOp_2?#layer0/lstm_cell_1/ReadVariableOp_3?'layer0/lstm_cell_1/split/ReadVariableOp?)layer0/lstm_cell_1/split_1/ReadVariableOp?layer0/while?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOpR
layer0/ShapeShapeinputs*
T0*
_output_shapes
:2
layer0/Shape?
layer0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
layer0/strided_slice/stack?
layer0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
layer0/strided_slice/stack_1?
layer0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
layer0/strided_slice/stack_2?
layer0/strided_sliceStridedSlicelayer0/Shape:output:0#layer0/strided_slice/stack:output:0%layer0/strided_slice/stack_1:output:0%layer0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer0/strided_slicek
layer0/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
layer0/zeros/mul/y?
layer0/zeros/mulMullayer0/strided_slice:output:0layer0/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
layer0/zeros/mulm
layer0/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
layer0/zeros/Less/y?
layer0/zeros/LessLesslayer0/zeros/mul:z:0layer0/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
layer0/zeros/Lessq
layer0/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
layer0/zeros/packed/1?
layer0/zeros/packedPacklayer0/strided_slice:output:0layer0/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
layer0/zeros/packedm
layer0/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
layer0/zeros/Const?
layer0/zerosFilllayer0/zeros/packed:output:0layer0/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
layer0/zeroso
layer0/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
layer0/zeros_1/mul/y?
layer0/zeros_1/mulMullayer0/strided_slice:output:0layer0/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
layer0/zeros_1/mulq
layer0/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
layer0/zeros_1/Less/y?
layer0/zeros_1/LessLesslayer0/zeros_1/mul:z:0layer0/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
layer0/zeros_1/Lessu
layer0/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
layer0/zeros_1/packed/1?
layer0/zeros_1/packedPacklayer0/strided_slice:output:0 layer0/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
layer0/zeros_1/packedq
layer0/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
layer0/zeros_1/Const?
layer0/zeros_1Filllayer0/zeros_1/packed:output:0layer0/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
layer0/zeros_1?
layer0/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
layer0/transpose/perm?
layer0/transpose	Transposeinputslayer0/transpose/perm:output:0*
T0*+
_output_shapes
:?????????P2
layer0/transposed
layer0/Shape_1Shapelayer0/transpose:y:0*
T0*
_output_shapes
:2
layer0/Shape_1?
layer0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
layer0/strided_slice_1/stack?
layer0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
layer0/strided_slice_1/stack_1?
layer0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
layer0/strided_slice_1/stack_2?
layer0/strided_slice_1StridedSlicelayer0/Shape_1:output:0%layer0/strided_slice_1/stack:output:0'layer0/strided_slice_1/stack_1:output:0'layer0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer0/strided_slice_1?
"layer0/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"layer0/TensorArrayV2/element_shape?
layer0/TensorArrayV2TensorListReserve+layer0/TensorArrayV2/element_shape:output:0layer0/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
layer0/TensorArrayV2?
<layer0/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   2>
<layer0/TensorArrayUnstack/TensorListFromTensor/element_shape?
.layer0/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlayer0/transpose:y:0Elayer0/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.layer0/TensorArrayUnstack/TensorListFromTensor?
layer0/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
layer0/strided_slice_2/stack?
layer0/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
layer0/strided_slice_2/stack_1?
layer0/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
layer0/strided_slice_2/stack_2?
layer0/strided_slice_2StridedSlicelayer0/transpose:y:0%layer0/strided_slice_2/stack:output:0'layer0/strided_slice_2/stack_1:output:0'layer0/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask2
layer0/strided_slice_2?
"layer0/lstm_cell_1/ones_like/ShapeShapelayer0/strided_slice_2:output:0*
T0*
_output_shapes
:2$
"layer0/lstm_cell_1/ones_like/Shape?
"layer0/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"layer0/lstm_cell_1/ones_like/Const?
layer0/lstm_cell_1/ones_likeFill+layer0/lstm_cell_1/ones_like/Shape:output:0+layer0/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????P2
layer0/lstm_cell_1/ones_like?
 layer0/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2"
 layer0/lstm_cell_1/dropout/Const?
layer0/lstm_cell_1/dropout/MulMul%layer0/lstm_cell_1/ones_like:output:0)layer0/lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????P2 
layer0/lstm_cell_1/dropout/Mul?
 layer0/lstm_cell_1/dropout/ShapeShape%layer0/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2"
 layer0/lstm_cell_1/dropout/Shape?
7layer0/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform)layer0/lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2???29
7layer0/lstm_cell_1/dropout/random_uniform/RandomUniform?
)layer0/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2+
)layer0/lstm_cell_1/dropout/GreaterEqual/y?
'layer0/lstm_cell_1/dropout/GreaterEqualGreaterEqual@layer0/lstm_cell_1/dropout/random_uniform/RandomUniform:output:02layer0/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2)
'layer0/lstm_cell_1/dropout/GreaterEqual?
layer0/lstm_cell_1/dropout/CastCast+layer0/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2!
layer0/lstm_cell_1/dropout/Cast?
 layer0/lstm_cell_1/dropout/Mul_1Mul"layer0/lstm_cell_1/dropout/Mul:z:0#layer0/lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????P2"
 layer0/lstm_cell_1/dropout/Mul_1?
"layer0/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2$
"layer0/lstm_cell_1/dropout_1/Const?
 layer0/lstm_cell_1/dropout_1/MulMul%layer0/lstm_cell_1/ones_like:output:0+layer0/lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????P2"
 layer0/lstm_cell_1/dropout_1/Mul?
"layer0/lstm_cell_1/dropout_1/ShapeShape%layer0/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2$
"layer0/lstm_cell_1/dropout_1/Shape?
9layer0/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform+layer0/lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2???2;
9layer0/lstm_cell_1/dropout_1/random_uniform/RandomUniform?
+layer0/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2-
+layer0/lstm_cell_1/dropout_1/GreaterEqual/y?
)layer0/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualBlayer0/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:04layer0/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2+
)layer0/lstm_cell_1/dropout_1/GreaterEqual?
!layer0/lstm_cell_1/dropout_1/CastCast-layer0/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2#
!layer0/lstm_cell_1/dropout_1/Cast?
"layer0/lstm_cell_1/dropout_1/Mul_1Mul$layer0/lstm_cell_1/dropout_1/Mul:z:0%layer0/lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????P2$
"layer0/lstm_cell_1/dropout_1/Mul_1?
"layer0/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2$
"layer0/lstm_cell_1/dropout_2/Const?
 layer0/lstm_cell_1/dropout_2/MulMul%layer0/lstm_cell_1/ones_like:output:0+layer0/lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????P2"
 layer0/lstm_cell_1/dropout_2/Mul?
"layer0/lstm_cell_1/dropout_2/ShapeShape%layer0/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2$
"layer0/lstm_cell_1/dropout_2/Shape?
9layer0/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform+layer0/lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2??k2;
9layer0/lstm_cell_1/dropout_2/random_uniform/RandomUniform?
+layer0/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2-
+layer0/lstm_cell_1/dropout_2/GreaterEqual/y?
)layer0/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualBlayer0/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:04layer0/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2+
)layer0/lstm_cell_1/dropout_2/GreaterEqual?
!layer0/lstm_cell_1/dropout_2/CastCast-layer0/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2#
!layer0/lstm_cell_1/dropout_2/Cast?
"layer0/lstm_cell_1/dropout_2/Mul_1Mul$layer0/lstm_cell_1/dropout_2/Mul:z:0%layer0/lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????P2$
"layer0/lstm_cell_1/dropout_2/Mul_1?
"layer0/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2$
"layer0/lstm_cell_1/dropout_3/Const?
 layer0/lstm_cell_1/dropout_3/MulMul%layer0/lstm_cell_1/ones_like:output:0+layer0/lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????P2"
 layer0/lstm_cell_1/dropout_3/Mul?
"layer0/lstm_cell_1/dropout_3/ShapeShape%layer0/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2$
"layer0/lstm_cell_1/dropout_3/Shape?
9layer0/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform+layer0/lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2??-2;
9layer0/lstm_cell_1/dropout_3/random_uniform/RandomUniform?
+layer0/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2-
+layer0/lstm_cell_1/dropout_3/GreaterEqual/y?
)layer0/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualBlayer0/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:04layer0/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2+
)layer0/lstm_cell_1/dropout_3/GreaterEqual?
!layer0/lstm_cell_1/dropout_3/CastCast-layer0/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2#
!layer0/lstm_cell_1/dropout_3/Cast?
"layer0/lstm_cell_1/dropout_3/Mul_1Mul$layer0/lstm_cell_1/dropout_3/Mul:z:0%layer0/lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????P2$
"layer0/lstm_cell_1/dropout_3/Mul_1?
$layer0/lstm_cell_1/ones_like_1/ShapeShapelayer0/zeros:output:0*
T0*
_output_shapes
:2&
$layer0/lstm_cell_1/ones_like_1/Shape?
$layer0/lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$layer0/lstm_cell_1/ones_like_1/Const?
layer0/lstm_cell_1/ones_like_1Fill-layer0/lstm_cell_1/ones_like_1/Shape:output:0-layer0/lstm_cell_1/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2 
layer0/lstm_cell_1/ones_like_1?
"layer0/lstm_cell_1/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2$
"layer0/lstm_cell_1/dropout_4/Const?
 layer0/lstm_cell_1/dropout_4/MulMul'layer0/lstm_cell_1/ones_like_1:output:0+layer0/lstm_cell_1/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2"
 layer0/lstm_cell_1/dropout_4/Mul?
"layer0/lstm_cell_1/dropout_4/ShapeShape'layer0/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2$
"layer0/lstm_cell_1/dropout_4/Shape?
9layer0/lstm_cell_1/dropout_4/random_uniform/RandomUniformRandomUniform+layer0/lstm_cell_1/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??p2;
9layer0/lstm_cell_1/dropout_4/random_uniform/RandomUniform?
+layer0/lstm_cell_1/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2-
+layer0/lstm_cell_1/dropout_4/GreaterEqual/y?
)layer0/lstm_cell_1/dropout_4/GreaterEqualGreaterEqualBlayer0/lstm_cell_1/dropout_4/random_uniform/RandomUniform:output:04layer0/lstm_cell_1/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2+
)layer0/lstm_cell_1/dropout_4/GreaterEqual?
!layer0/lstm_cell_1/dropout_4/CastCast-layer0/lstm_cell_1/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2#
!layer0/lstm_cell_1/dropout_4/Cast?
"layer0/lstm_cell_1/dropout_4/Mul_1Mul$layer0/lstm_cell_1/dropout_4/Mul:z:0%layer0/lstm_cell_1/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2$
"layer0/lstm_cell_1/dropout_4/Mul_1?
"layer0/lstm_cell_1/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2$
"layer0/lstm_cell_1/dropout_5/Const?
 layer0/lstm_cell_1/dropout_5/MulMul'layer0/lstm_cell_1/ones_like_1:output:0+layer0/lstm_cell_1/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2"
 layer0/lstm_cell_1/dropout_5/Mul?
"layer0/lstm_cell_1/dropout_5/ShapeShape'layer0/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2$
"layer0/lstm_cell_1/dropout_5/Shape?
9layer0/lstm_cell_1/dropout_5/random_uniform/RandomUniformRandomUniform+layer0/lstm_cell_1/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2;
9layer0/lstm_cell_1/dropout_5/random_uniform/RandomUniform?
+layer0/lstm_cell_1/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2-
+layer0/lstm_cell_1/dropout_5/GreaterEqual/y?
)layer0/lstm_cell_1/dropout_5/GreaterEqualGreaterEqualBlayer0/lstm_cell_1/dropout_5/random_uniform/RandomUniform:output:04layer0/lstm_cell_1/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2+
)layer0/lstm_cell_1/dropout_5/GreaterEqual?
!layer0/lstm_cell_1/dropout_5/CastCast-layer0/lstm_cell_1/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2#
!layer0/lstm_cell_1/dropout_5/Cast?
"layer0/lstm_cell_1/dropout_5/Mul_1Mul$layer0/lstm_cell_1/dropout_5/Mul:z:0%layer0/lstm_cell_1/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2$
"layer0/lstm_cell_1/dropout_5/Mul_1?
"layer0/lstm_cell_1/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2$
"layer0/lstm_cell_1/dropout_6/Const?
 layer0/lstm_cell_1/dropout_6/MulMul'layer0/lstm_cell_1/ones_like_1:output:0+layer0/lstm_cell_1/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2"
 layer0/lstm_cell_1/dropout_6/Mul?
"layer0/lstm_cell_1/dropout_6/ShapeShape'layer0/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2$
"layer0/lstm_cell_1/dropout_6/Shape?
9layer0/lstm_cell_1/dropout_6/random_uniform/RandomUniformRandomUniform+layer0/lstm_cell_1/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??w2;
9layer0/lstm_cell_1/dropout_6/random_uniform/RandomUniform?
+layer0/lstm_cell_1/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2-
+layer0/lstm_cell_1/dropout_6/GreaterEqual/y?
)layer0/lstm_cell_1/dropout_6/GreaterEqualGreaterEqualBlayer0/lstm_cell_1/dropout_6/random_uniform/RandomUniform:output:04layer0/lstm_cell_1/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2+
)layer0/lstm_cell_1/dropout_6/GreaterEqual?
!layer0/lstm_cell_1/dropout_6/CastCast-layer0/lstm_cell_1/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2#
!layer0/lstm_cell_1/dropout_6/Cast?
"layer0/lstm_cell_1/dropout_6/Mul_1Mul$layer0/lstm_cell_1/dropout_6/Mul:z:0%layer0/lstm_cell_1/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2$
"layer0/lstm_cell_1/dropout_6/Mul_1?
"layer0/lstm_cell_1/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2$
"layer0/lstm_cell_1/dropout_7/Const?
 layer0/lstm_cell_1/dropout_7/MulMul'layer0/lstm_cell_1/ones_like_1:output:0+layer0/lstm_cell_1/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2"
 layer0/lstm_cell_1/dropout_7/Mul?
"layer0/lstm_cell_1/dropout_7/ShapeShape'layer0/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2$
"layer0/lstm_cell_1/dropout_7/Shape?
9layer0/lstm_cell_1/dropout_7/random_uniform/RandomUniformRandomUniform+layer0/lstm_cell_1/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2;
9layer0/lstm_cell_1/dropout_7/random_uniform/RandomUniform?
+layer0/lstm_cell_1/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2-
+layer0/lstm_cell_1/dropout_7/GreaterEqual/y?
)layer0/lstm_cell_1/dropout_7/GreaterEqualGreaterEqualBlayer0/lstm_cell_1/dropout_7/random_uniform/RandomUniform:output:04layer0/lstm_cell_1/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2+
)layer0/lstm_cell_1/dropout_7/GreaterEqual?
!layer0/lstm_cell_1/dropout_7/CastCast-layer0/lstm_cell_1/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2#
!layer0/lstm_cell_1/dropout_7/Cast?
"layer0/lstm_cell_1/dropout_7/Mul_1Mul$layer0/lstm_cell_1/dropout_7/Mul:z:0%layer0/lstm_cell_1/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2$
"layer0/lstm_cell_1/dropout_7/Mul_1?
layer0/lstm_cell_1/mulMullayer0/strided_slice_2:output:0$layer0/lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
layer0/lstm_cell_1/mul?
layer0/lstm_cell_1/mul_1Mullayer0/strided_slice_2:output:0&layer0/lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
layer0/lstm_cell_1/mul_1?
layer0/lstm_cell_1/mul_2Mullayer0/strided_slice_2:output:0&layer0/lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
layer0/lstm_cell_1/mul_2?
layer0/lstm_cell_1/mul_3Mullayer0/strided_slice_2:output:0&layer0/lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
layer0/lstm_cell_1/mul_3v
layer0/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
layer0/lstm_cell_1/Const?
"layer0/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"layer0/lstm_cell_1/split/split_dim?
'layer0/lstm_cell_1/split/ReadVariableOpReadVariableOp0layer0_lstm_cell_1_split_readvariableop_resource*
_output_shapes
:	P? *
dtype02)
'layer0/lstm_cell_1/split/ReadVariableOp?
layer0/lstm_cell_1/splitSplit+layer0/lstm_cell_1/split/split_dim:output:0/layer0/lstm_cell_1/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	P?:	P?:	P?:	P?*
	num_split2
layer0/lstm_cell_1/split?
layer0/lstm_cell_1/MatMulMatMullayer0/lstm_cell_1/mul:z:0!layer0/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/MatMul?
layer0/lstm_cell_1/MatMul_1MatMullayer0/lstm_cell_1/mul_1:z:0!layer0/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/MatMul_1?
layer0/lstm_cell_1/MatMul_2MatMullayer0/lstm_cell_1/mul_2:z:0!layer0/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/MatMul_2?
layer0/lstm_cell_1/MatMul_3MatMullayer0/lstm_cell_1/mul_3:z:0!layer0/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/MatMul_3z
layer0/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
layer0/lstm_cell_1/Const_1?
$layer0/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$layer0/lstm_cell_1/split_1/split_dim?
)layer0/lstm_cell_1/split_1/ReadVariableOpReadVariableOp2layer0_lstm_cell_1_split_1_readvariableop_resource*
_output_shapes	
:? *
dtype02+
)layer0/lstm_cell_1/split_1/ReadVariableOp?
layer0/lstm_cell_1/split_1Split-layer0/lstm_cell_1/split_1/split_dim:output:01layer0/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
layer0/lstm_cell_1/split_1?
layer0/lstm_cell_1/BiasAddBiasAdd#layer0/lstm_cell_1/MatMul:product:0#layer0/lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/BiasAdd?
layer0/lstm_cell_1/BiasAdd_1BiasAdd%layer0/lstm_cell_1/MatMul_1:product:0#layer0/lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/BiasAdd_1?
layer0/lstm_cell_1/BiasAdd_2BiasAdd%layer0/lstm_cell_1/MatMul_2:product:0#layer0/lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/BiasAdd_2?
layer0/lstm_cell_1/BiasAdd_3BiasAdd%layer0/lstm_cell_1/MatMul_3:product:0#layer0/lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/BiasAdd_3?
layer0/lstm_cell_1/mul_4Mullayer0/zeros:output:0&layer0/lstm_cell_1/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/mul_4?
layer0/lstm_cell_1/mul_5Mullayer0/zeros:output:0&layer0/lstm_cell_1/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/mul_5?
layer0/lstm_cell_1/mul_6Mullayer0/zeros:output:0&layer0/lstm_cell_1/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/mul_6?
layer0/lstm_cell_1/mul_7Mullayer0/zeros:output:0&layer0/lstm_cell_1/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/mul_7?
!layer0/lstm_cell_1/ReadVariableOpReadVariableOp*layer0_lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02#
!layer0/lstm_cell_1/ReadVariableOp?
&layer0/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&layer0/lstm_cell_1/strided_slice/stack?
(layer0/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2*
(layer0/lstm_cell_1/strided_slice/stack_1?
(layer0/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(layer0/lstm_cell_1/strided_slice/stack_2?
 layer0/lstm_cell_1/strided_sliceStridedSlice)layer0/lstm_cell_1/ReadVariableOp:value:0/layer0/lstm_cell_1/strided_slice/stack:output:01layer0/lstm_cell_1/strided_slice/stack_1:output:01layer0/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 layer0/lstm_cell_1/strided_slice?
layer0/lstm_cell_1/MatMul_4MatMullayer0/lstm_cell_1/mul_4:z:0)layer0/lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/MatMul_4?
layer0/lstm_cell_1/addAddV2#layer0/lstm_cell_1/BiasAdd:output:0%layer0/lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/add?
layer0/lstm_cell_1/SigmoidSigmoidlayer0/lstm_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/Sigmoid?
#layer0/lstm_cell_1/ReadVariableOp_1ReadVariableOp*layer0_lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02%
#layer0/lstm_cell_1/ReadVariableOp_1?
(layer0/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2*
(layer0/lstm_cell_1/strided_slice_1/stack?
*layer0/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2,
*layer0/lstm_cell_1/strided_slice_1/stack_1?
*layer0/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*layer0/lstm_cell_1/strided_slice_1/stack_2?
"layer0/lstm_cell_1/strided_slice_1StridedSlice+layer0/lstm_cell_1/ReadVariableOp_1:value:01layer0/lstm_cell_1/strided_slice_1/stack:output:03layer0/lstm_cell_1/strided_slice_1/stack_1:output:03layer0/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2$
"layer0/lstm_cell_1/strided_slice_1?
layer0/lstm_cell_1/MatMul_5MatMullayer0/lstm_cell_1/mul_5:z:0+layer0/lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/MatMul_5?
layer0/lstm_cell_1/add_1AddV2%layer0/lstm_cell_1/BiasAdd_1:output:0%layer0/lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/add_1?
layer0/lstm_cell_1/Sigmoid_1Sigmoidlayer0/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/Sigmoid_1?
layer0/lstm_cell_1/mul_8Mul layer0/lstm_cell_1/Sigmoid_1:y:0layer0/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/mul_8?
#layer0/lstm_cell_1/ReadVariableOp_2ReadVariableOp*layer0_lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02%
#layer0/lstm_cell_1/ReadVariableOp_2?
(layer0/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2*
(layer0/lstm_cell_1/strided_slice_2/stack?
*layer0/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2,
*layer0/lstm_cell_1/strided_slice_2/stack_1?
*layer0/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*layer0/lstm_cell_1/strided_slice_2/stack_2?
"layer0/lstm_cell_1/strided_slice_2StridedSlice+layer0/lstm_cell_1/ReadVariableOp_2:value:01layer0/lstm_cell_1/strided_slice_2/stack:output:03layer0/lstm_cell_1/strided_slice_2/stack_1:output:03layer0/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2$
"layer0/lstm_cell_1/strided_slice_2?
layer0/lstm_cell_1/MatMul_6MatMullayer0/lstm_cell_1/mul_6:z:0+layer0/lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/MatMul_6?
layer0/lstm_cell_1/add_2AddV2%layer0/lstm_cell_1/BiasAdd_2:output:0%layer0/lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/add_2?
layer0/lstm_cell_1/TanhTanhlayer0/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/Tanh?
layer0/lstm_cell_1/mul_9Mullayer0/lstm_cell_1/Sigmoid:y:0layer0/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/mul_9?
layer0/lstm_cell_1/add_3AddV2layer0/lstm_cell_1/mul_8:z:0layer0/lstm_cell_1/mul_9:z:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/add_3?
#layer0/lstm_cell_1/ReadVariableOp_3ReadVariableOp*layer0_lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02%
#layer0/lstm_cell_1/ReadVariableOp_3?
(layer0/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2*
(layer0/lstm_cell_1/strided_slice_3/stack?
*layer0/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*layer0/lstm_cell_1/strided_slice_3/stack_1?
*layer0/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*layer0/lstm_cell_1/strided_slice_3/stack_2?
"layer0/lstm_cell_1/strided_slice_3StridedSlice+layer0/lstm_cell_1/ReadVariableOp_3:value:01layer0/lstm_cell_1/strided_slice_3/stack:output:03layer0/lstm_cell_1/strided_slice_3/stack_1:output:03layer0/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2$
"layer0/lstm_cell_1/strided_slice_3?
layer0/lstm_cell_1/MatMul_7MatMullayer0/lstm_cell_1/mul_7:z:0+layer0/lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/MatMul_7?
layer0/lstm_cell_1/add_4AddV2%layer0/lstm_cell_1/BiasAdd_3:output:0%layer0/lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/add_4?
layer0/lstm_cell_1/Sigmoid_2Sigmoidlayer0/lstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/Sigmoid_2?
layer0/lstm_cell_1/Tanh_1Tanhlayer0/lstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/Tanh_1?
layer0/lstm_cell_1/mul_10Mul layer0/lstm_cell_1/Sigmoid_2:y:0layer0/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
layer0/lstm_cell_1/mul_10?
$layer0/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2&
$layer0/TensorArrayV2_1/element_shape?
layer0/TensorArrayV2_1TensorListReserve-layer0/TensorArrayV2_1/element_shape:output:0layer0/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
layer0/TensorArrayV2_1\
layer0/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
layer0/time?
layer0/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
layer0/while/maximum_iterationsx
layer0/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
layer0/while/loop_counter?
layer0/whileWhile"layer0/while/loop_counter:output:0(layer0/while/maximum_iterations:output:0layer0/time:output:0layer0/TensorArrayV2_1:handle:0layer0/zeros:output:0layer0/zeros_1:output:0layer0/strided_slice_1:output:0>layer0/TensorArrayUnstack/TensorListFromTensor:output_handle:00layer0_lstm_cell_1_split_readvariableop_resource2layer0_lstm_cell_1_split_1_readvariableop_resource*layer0_lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*$
bodyR
layer0_while_body_349208*$
condR
layer0_while_cond_349207*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
layer0/while?
7layer0/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7layer0/TensorArrayV2Stack/TensorListStack/element_shape?
)layer0/TensorArrayV2Stack/TensorListStackTensorListStacklayer0/while:output:3@layer0/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02+
)layer0/TensorArrayV2Stack/TensorListStack?
layer0/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
layer0/strided_slice_3/stack?
layer0/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
layer0/strided_slice_3/stack_1?
layer0/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
layer0/strided_slice_3/stack_2?
layer0/strided_slice_3StridedSlice2layer0/TensorArrayV2Stack/TensorListStack:tensor:0%layer0/strided_slice_3/stack:output:0'layer0/strided_slice_3/stack_1:output:0'layer0/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
layer0/strided_slice_3?
layer0/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
layer0/transpose_1/perm?
layer0/transpose_1	Transpose2layer0/TensorArrayV2Stack/TensorListStack:tensor:0 layer0/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
layer0/transpose_1t
layer0/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
layer0/runtime?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_10/MatMul/ReadVariableOp?
dense_10/MatMulMatMullayer0/strided_slice_3:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_10/MatMul?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_10/BiasAddt
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_10/Relu?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMuldense_10/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
output/Softmax?
IdentityIdentityoutput/Softmax:softmax:0 ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp"^layer0/lstm_cell_1/ReadVariableOp$^layer0/lstm_cell_1/ReadVariableOp_1$^layer0/lstm_cell_1/ReadVariableOp_2$^layer0/lstm_cell_1/ReadVariableOp_3(^layer0/lstm_cell_1/split/ReadVariableOp*^layer0/lstm_cell_1/split_1/ReadVariableOp^layer0/while^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????P:::::::2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2F
!layer0/lstm_cell_1/ReadVariableOp!layer0/lstm_cell_1/ReadVariableOp2J
#layer0/lstm_cell_1/ReadVariableOp_1#layer0/lstm_cell_1/ReadVariableOp_12J
#layer0/lstm_cell_1/ReadVariableOp_2#layer0/lstm_cell_1/ReadVariableOp_22J
#layer0/lstm_cell_1/ReadVariableOp_3#layer0/lstm_cell_1/ReadVariableOp_32R
'layer0/lstm_cell_1/split/ReadVariableOp'layer0/lstm_cell_1/split/ReadVariableOp2V
)layer0/lstm_cell_1/split_1/ReadVariableOp)layer0/lstm_cell_1/split_1/ReadVariableOp2
layer0/whilelayer0/while2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
??
?
B__inference_layer0_layer_call_and_return_conditional_losses_350112
inputs_0-
)lstm_cell_1_split_readvariableop_resource/
+lstm_cell_1_split_1_readvariableop_resource'
#lstm_cell_1_readvariableop_resource
identity??lstm_cell_1/ReadVariableOp?lstm_cell_1/ReadVariableOp_1?lstm_cell_1/ReadVariableOp_2?lstm_cell_1/ReadVariableOp_3? lstm_cell_1/split/ReadVariableOp?"lstm_cell_1/split_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????P2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask2
strided_slice_2?
lstm_cell_1/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like/Shape
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_1/ones_like/Const?
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/ones_like{
lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
lstm_cell_1/dropout/Const?
lstm_cell_1/dropout/MulMullstm_cell_1/ones_like:output:0"lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/dropout/Mul?
lstm_cell_1/dropout/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout/Shape?
0lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2???22
0lstm_cell_1/dropout/random_uniform/RandomUniform?
"lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2$
"lstm_cell_1/dropout/GreaterEqual/y?
 lstm_cell_1/dropout/GreaterEqualGreaterEqual9lstm_cell_1/dropout/random_uniform/RandomUniform:output:0+lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2"
 lstm_cell_1/dropout/GreaterEqual?
lstm_cell_1/dropout/CastCast$lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2
lstm_cell_1/dropout/Cast?
lstm_cell_1/dropout/Mul_1Mullstm_cell_1/dropout/Mul:z:0lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/dropout/Mul_1
lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
lstm_cell_1/dropout_1/Const?
lstm_cell_1/dropout_1/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_1/Mul?
lstm_cell_1/dropout_1/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_1/Shape?
2lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2??h24
2lstm_cell_1/dropout_1/random_uniform/RandomUniform?
$lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_1/dropout_1/GreaterEqual/y?
"lstm_cell_1/dropout_1/GreaterEqualGreaterEqual;lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2$
"lstm_cell_1/dropout_1/GreaterEqual?
lstm_cell_1/dropout_1/CastCast&lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_1/Cast?
lstm_cell_1/dropout_1/Mul_1Mullstm_cell_1/dropout_1/Mul:z:0lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_1/Mul_1
lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
lstm_cell_1/dropout_2/Const?
lstm_cell_1/dropout_2/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_2/Mul?
lstm_cell_1/dropout_2/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_2/Shape?
2lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2?̾24
2lstm_cell_1/dropout_2/random_uniform/RandomUniform?
$lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_1/dropout_2/GreaterEqual/y?
"lstm_cell_1/dropout_2/GreaterEqualGreaterEqual;lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2$
"lstm_cell_1/dropout_2/GreaterEqual?
lstm_cell_1/dropout_2/CastCast&lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_2/Cast?
lstm_cell_1/dropout_2/Mul_1Mullstm_cell_1/dropout_2/Mul:z:0lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_2/Mul_1
lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
lstm_cell_1/dropout_3/Const?
lstm_cell_1/dropout_3/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_3/Mul?
lstm_cell_1/dropout_3/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_3/Shape?
2lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2???24
2lstm_cell_1/dropout_3/random_uniform/RandomUniform?
$lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_1/dropout_3/GreaterEqual/y?
"lstm_cell_1/dropout_3/GreaterEqualGreaterEqual;lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2$
"lstm_cell_1/dropout_3/GreaterEqual?
lstm_cell_1/dropout_3/CastCast&lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_3/Cast?
lstm_cell_1/dropout_3/Mul_1Mullstm_cell_1/dropout_3/Mul:z:0lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/dropout_3/Mul_1|
lstm_cell_1/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like_1/Shape?
lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_1/ones_like_1/Const?
lstm_cell_1/ones_like_1Fill&lstm_cell_1/ones_like_1/Shape:output:0&lstm_cell_1/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/ones_like_1
lstm_cell_1/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
lstm_cell_1/dropout_4/Const?
lstm_cell_1/dropout_4/MulMul lstm_cell_1/ones_like_1:output:0$lstm_cell_1/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/dropout_4/Mul?
lstm_cell_1/dropout_4/ShapeShape lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_4/Shape?
2lstm_cell_1/dropout_4/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ɵ24
2lstm_cell_1/dropout_4/random_uniform/RandomUniform?
$lstm_cell_1/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_1/dropout_4/GreaterEqual/y?
"lstm_cell_1/dropout_4/GreaterEqualGreaterEqual;lstm_cell_1/dropout_4/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_cell_1/dropout_4/GreaterEqual?
lstm_cell_1/dropout_4/CastCast&lstm_cell_1/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell_1/dropout_4/Cast?
lstm_cell_1/dropout_4/Mul_1Mullstm_cell_1/dropout_4/Mul:z:0lstm_cell_1/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/dropout_4/Mul_1
lstm_cell_1/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
lstm_cell_1/dropout_5/Const?
lstm_cell_1/dropout_5/MulMul lstm_cell_1/ones_like_1:output:0$lstm_cell_1/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/dropout_5/Mul?
lstm_cell_1/dropout_5/ShapeShape lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_5/Shape?
2lstm_cell_1/dropout_5/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???24
2lstm_cell_1/dropout_5/random_uniform/RandomUniform?
$lstm_cell_1/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_1/dropout_5/GreaterEqual/y?
"lstm_cell_1/dropout_5/GreaterEqualGreaterEqual;lstm_cell_1/dropout_5/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_cell_1/dropout_5/GreaterEqual?
lstm_cell_1/dropout_5/CastCast&lstm_cell_1/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell_1/dropout_5/Cast?
lstm_cell_1/dropout_5/Mul_1Mullstm_cell_1/dropout_5/Mul:z:0lstm_cell_1/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/dropout_5/Mul_1
lstm_cell_1/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
lstm_cell_1/dropout_6/Const?
lstm_cell_1/dropout_6/MulMul lstm_cell_1/ones_like_1:output:0$lstm_cell_1/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/dropout_6/Mul?
lstm_cell_1/dropout_6/ShapeShape lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_6/Shape?
2lstm_cell_1/dropout_6/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???24
2lstm_cell_1/dropout_6/random_uniform/RandomUniform?
$lstm_cell_1/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_1/dropout_6/GreaterEqual/y?
"lstm_cell_1/dropout_6/GreaterEqualGreaterEqual;lstm_cell_1/dropout_6/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_cell_1/dropout_6/GreaterEqual?
lstm_cell_1/dropout_6/CastCast&lstm_cell_1/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell_1/dropout_6/Cast?
lstm_cell_1/dropout_6/Mul_1Mullstm_cell_1/dropout_6/Mul:z:0lstm_cell_1/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/dropout_6/Mul_1
lstm_cell_1/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
lstm_cell_1/dropout_7/Const?
lstm_cell_1/dropout_7/MulMul lstm_cell_1/ones_like_1:output:0$lstm_cell_1/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/dropout_7/Mul?
lstm_cell_1/dropout_7/ShapeShape lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_7/Shape?
2lstm_cell_1/dropout_7/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???24
2lstm_cell_1/dropout_7/random_uniform/RandomUniform?
$lstm_cell_1/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_1/dropout_7/GreaterEqual/y?
"lstm_cell_1/dropout_7/GreaterEqualGreaterEqual;lstm_cell_1/dropout_7/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_cell_1/dropout_7/GreaterEqual?
lstm_cell_1/dropout_7/CastCast&lstm_cell_1/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell_1/dropout_7/Cast?
lstm_cell_1/dropout_7/Mul_1Mullstm_cell_1/dropout_7/Mul:z:0lstm_cell_1/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/dropout_7/Mul_1?
lstm_cell_1/mulMulstrided_slice_2:output:0lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/mul?
lstm_cell_1/mul_1Mulstrided_slice_2:output:0lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/mul_1?
lstm_cell_1/mul_2Mulstrided_slice_2:output:0lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/mul_2?
lstm_cell_1/mul_3Mulstrided_slice_2:output:0lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/mul_3h
lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dim?
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource*
_output_shapes
:	P? *
dtype02"
 lstm_cell_1/split/ReadVariableOp?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	P?:	P?:	P?:	P?*
	num_split2
lstm_cell_1/split?
lstm_cell_1/MatMulMatMullstm_cell_1/mul:z:0lstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul?
lstm_cell_1/MatMul_1MatMullstm_cell_1/mul_1:z:0lstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_1?
lstm_cell_1/MatMul_2MatMullstm_cell_1/mul_2:z:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_2?
lstm_cell_1/MatMul_3MatMullstm_cell_1/mul_3:z:0lstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_3l
lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const_1?
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_1/split_1/split_dim?
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes	
:? *
dtype02$
"lstm_cell_1/split_1/ReadVariableOp?
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell_1/split_1?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/BiasAdd?
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_1/BiasAdd_1?
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_1/BiasAdd_2?
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_1/BiasAdd_3?
lstm_cell_1/mul_4Mulzeros:output:0lstm_cell_1/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_4?
lstm_cell_1/mul_5Mulzeros:output:0lstm_cell_1/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_5?
lstm_cell_1/mul_6Mulzeros:output:0lstm_cell_1/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_6?
lstm_cell_1/mul_7Mulzeros:output:0lstm_cell_1/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_7?
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
lstm_cell_1/ReadVariableOp?
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_1/strided_slice/stack?
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice/stack_1?
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_1/strided_slice/stack_2?
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_1/strided_slice?
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul_4:z:0"lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_4?
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add}
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Sigmoid?
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
lstm_cell_1/ReadVariableOp_1?
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_1/stack?
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#lstm_cell_1/strided_slice_1/stack_1?
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_1/stack_2?
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_1?
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_5:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_5?
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add_1?
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Sigmoid_1?
lstm_cell_1/mul_8Mullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_8?
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
lstm_cell_1/ReadVariableOp_2?
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_2/stack?
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#lstm_cell_1/strided_slice_2/stack_1?
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_2/stack_2?
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_2?
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_6:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_6?
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add_2v
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Tanh?
lstm_cell_1/mul_9Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_9?
lstm_cell_1/add_3AddV2lstm_cell_1/mul_8:z:0lstm_cell_1/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add_3?
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
lstm_cell_1/ReadVariableOp_3?
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_3/stack?
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_1/strided_slice_3/stack_1?
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_3/stack_2?
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_3?
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_7:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_7?
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add_4?
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Sigmoid_2z
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Tanh_1?
lstm_cell_1/mul_10Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_349912*
condR
while_cond_349911*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????P:::28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????P
"
_user_specified_name
inputs/0
?
?
while_cond_350571
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_350571___redundant_placeholder04
0while_while_cond_350571___redundant_placeholder14
0while_while_cond_350571___redundant_placeholder24
0while_while_cond_350571___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
B__inference_layer0_layer_call_and_return_conditional_losses_351027

inputs-
)lstm_cell_1_split_readvariableop_resource/
+lstm_cell_1_split_1_readvariableop_resource'
#lstm_cell_1_readvariableop_resource
identity??lstm_cell_1/ReadVariableOp?lstm_cell_1/ReadVariableOp_1?lstm_cell_1/ReadVariableOp_2?lstm_cell_1/ReadVariableOp_3? lstm_cell_1/split/ReadVariableOp?"lstm_cell_1/split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????P2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask2
strided_slice_2?
lstm_cell_1/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like/Shape
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_1/ones_like/Const?
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/ones_like|
lstm_cell_1/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like_1/Shape?
lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_1/ones_like_1/Const?
lstm_cell_1/ones_like_1Fill&lstm_cell_1/ones_like_1/Shape:output:0&lstm_cell_1/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/ones_like_1?
lstm_cell_1/mulMulstrided_slice_2:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/mul?
lstm_cell_1/mul_1Mulstrided_slice_2:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/mul_1?
lstm_cell_1/mul_2Mulstrided_slice_2:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/mul_2?
lstm_cell_1/mul_3Mulstrided_slice_2:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/mul_3h
lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dim?
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource*
_output_shapes
:	P? *
dtype02"
 lstm_cell_1/split/ReadVariableOp?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	P?:	P?:	P?:	P?*
	num_split2
lstm_cell_1/split?
lstm_cell_1/MatMulMatMullstm_cell_1/mul:z:0lstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul?
lstm_cell_1/MatMul_1MatMullstm_cell_1/mul_1:z:0lstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_1?
lstm_cell_1/MatMul_2MatMullstm_cell_1/mul_2:z:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_2?
lstm_cell_1/MatMul_3MatMullstm_cell_1/mul_3:z:0lstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_3l
lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const_1?
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_1/split_1/split_dim?
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes	
:? *
dtype02$
"lstm_cell_1/split_1/ReadVariableOp?
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell_1/split_1?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/BiasAdd?
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_1/BiasAdd_1?
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_1/BiasAdd_2?
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_1/BiasAdd_3?
lstm_cell_1/mul_4Mulzeros:output:0 lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_4?
lstm_cell_1/mul_5Mulzeros:output:0 lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_5?
lstm_cell_1/mul_6Mulzeros:output:0 lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_6?
lstm_cell_1/mul_7Mulzeros:output:0 lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_7?
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
lstm_cell_1/ReadVariableOp?
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_1/strided_slice/stack?
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice/stack_1?
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_1/strided_slice/stack_2?
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_1/strided_slice?
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul_4:z:0"lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_4?
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add}
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Sigmoid?
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
lstm_cell_1/ReadVariableOp_1?
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_1/stack?
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#lstm_cell_1/strided_slice_1/stack_1?
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_1/stack_2?
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_1?
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_5:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_5?
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add_1?
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Sigmoid_1?
lstm_cell_1/mul_8Mullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_8?
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
lstm_cell_1/ReadVariableOp_2?
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_2/stack?
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#lstm_cell_1/strided_slice_2/stack_1?
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_2/stack_2?
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_2?
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_6:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_6?
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add_2v
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Tanh?
lstm_cell_1/mul_9Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_9?
lstm_cell_1/add_3AddV2lstm_cell_1/mul_8:z:0lstm_cell_1/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add_3?
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
lstm_cell_1/ReadVariableOp_3?
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_3/stack?
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_1/strided_slice_3/stack_1?
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_3/stack_2?
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_3?
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_7:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_7?
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add_4?
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Sigmoid_2z
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Tanh_1?
lstm_cell_1/mul_10Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_350891*
condR
while_cond_350890*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????P:::28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?D
?
B__inference_layer0_layer_call_and_return_conditional_losses_348027

inputs
lstm_cell_1_347945
lstm_cell_1_347947
lstm_cell_1_347949
identity??#lstm_cell_1/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????P2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask2
strided_slice_2?
#lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1_347945lstm_cell_1_347947lstm_cell_1_347949*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_3475802%
#lstm_cell_1/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1_347945lstm_cell_1_347947lstm_cell_1_347949*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_347958*
condR
while_cond_347957*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_1/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????P:::2J
#lstm_cell_1/StatefulPartitionedCall#lstm_cell_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????P
 
_user_specified_nameinputs
?	
?
D__inference_dense_10_layer_call_and_return_conditional_losses_351060

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_348672
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_348672___redundant_placeholder04
0while_while_cond_348672___redundant_placeholder14
0while_while_cond_348672___redundant_placeholder24
0while_while_cond_348672___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?	
?
B__inference_output_layer_call_and_return_conditional_losses_351080

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_layer0_layer_call_fn_350389
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer0_layer_call_and_return_conditional_losses_3481592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????P:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????P
"
_user_specified_name
inputs/0
?
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_348979

inputs
layer0_348961
layer0_348963
layer0_348965
dense_10_348968
dense_10_348970
output_348973
output_348975
identity?? dense_10/StatefulPartitionedCall?layer0/StatefulPartitionedCall?output/StatefulPartitionedCall?
layer0/StatefulPartitionedCallStatefulPartitionedCallinputslayer0_348961layer0_348963layer0_348965*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer0_layer_call_and_return_conditional_losses_3488092 
layer0/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall'layer0/StatefulPartitionedCall:output:0dense_10_348968dense_10_348970*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_3488502"
 dense_10/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0output_348973output_348975*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_3488772 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall^layer0/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????P:::::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2@
layer0/StatefulPartitionedCalllayer0/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
??
?
while_body_348354
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
1while_lstm_cell_1_split_readvariableop_resource_07
3while_lstm_cell_1_split_1_readvariableop_resource_0/
+while_lstm_cell_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
/while_lstm_cell_1_split_readvariableop_resource5
1while_lstm_cell_1_split_1_readvariableop_resource-
)while_lstm_cell_1_readvariableop_resource?? while/lstm_cell_1/ReadVariableOp?"while/lstm_cell_1/ReadVariableOp_1?"while/lstm_cell_1/ReadVariableOp_2?"while/lstm_cell_1/ReadVariableOp_3?&while/lstm_cell_1/split/ReadVariableOp?(while/lstm_cell_1/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
!while/lstm_cell_1/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/ones_like/Shape?
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell_1/ones_like/Const?
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/ones_like?
while/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2!
while/lstm_cell_1/dropout/Const?
while/lstm_cell_1/dropout/MulMul$while/lstm_cell_1/ones_like:output:0(while/lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/dropout/Mul?
while/lstm_cell_1/dropout/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_1/dropout/Shape?
6while/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2??28
6while/lstm_cell_1/dropout/random_uniform/RandomUniform?
(while/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2*
(while/lstm_cell_1/dropout/GreaterEqual/y?
&while/lstm_cell_1/dropout/GreaterEqualGreaterEqual?while/lstm_cell_1/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2(
&while/lstm_cell_1/dropout/GreaterEqual?
while/lstm_cell_1/dropout/CastCast*while/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2 
while/lstm_cell_1/dropout/Cast?
while/lstm_cell_1/dropout/Mul_1Mul!while/lstm_cell_1/dropout/Mul:z:0"while/lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????P2!
while/lstm_cell_1/dropout/Mul_1?
!while/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!while/lstm_cell_1/dropout_1/Const?
while/lstm_cell_1/dropout_1/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????P2!
while/lstm_cell_1/dropout_1/Mul?
!while/lstm_cell_1/dropout_1/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_1/Shape?
8while/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2Ԏj2:
8while/lstm_cell_1/dropout_1/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_1/dropout_1/GreaterEqual/y?
(while/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2*
(while/lstm_cell_1/dropout_1/GreaterEqual?
 while/lstm_cell_1/dropout_1/CastCast,while/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2"
 while/lstm_cell_1/dropout_1/Cast?
!while/lstm_cell_1/dropout_1/Mul_1Mul#while/lstm_cell_1/dropout_1/Mul:z:0$while/lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????P2#
!while/lstm_cell_1/dropout_1/Mul_1?
!while/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!while/lstm_cell_1/dropout_2/Const?
while/lstm_cell_1/dropout_2/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????P2!
while/lstm_cell_1/dropout_2/Mul?
!while/lstm_cell_1/dropout_2/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_2/Shape?
8while/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2??[2:
8while/lstm_cell_1/dropout_2/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_1/dropout_2/GreaterEqual/y?
(while/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2*
(while/lstm_cell_1/dropout_2/GreaterEqual?
 while/lstm_cell_1/dropout_2/CastCast,while/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2"
 while/lstm_cell_1/dropout_2/Cast?
!while/lstm_cell_1/dropout_2/Mul_1Mul#while/lstm_cell_1/dropout_2/Mul:z:0$while/lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????P2#
!while/lstm_cell_1/dropout_2/Mul_1?
!while/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!while/lstm_cell_1/dropout_3/Const?
while/lstm_cell_1/dropout_3/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????P2!
while/lstm_cell_1/dropout_3/Mul?
!while/lstm_cell_1/dropout_3/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_3/Shape?
8while/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_1/dropout_3/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_1/dropout_3/GreaterEqual/y?
(while/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2*
(while/lstm_cell_1/dropout_3/GreaterEqual?
 while/lstm_cell_1/dropout_3/CastCast,while/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2"
 while/lstm_cell_1/dropout_3/Cast?
!while/lstm_cell_1/dropout_3/Mul_1Mul#while/lstm_cell_1/dropout_3/Mul:z:0$while/lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????P2#
!while/lstm_cell_1/dropout_3/Mul_1?
#while/lstm_cell_1/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_1/ones_like_1/Shape?
#while/lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#while/lstm_cell_1/ones_like_1/Const?
while/lstm_cell_1/ones_like_1Fill,while/lstm_cell_1/ones_like_1/Shape:output:0,while/lstm_cell_1/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/ones_like_1?
!while/lstm_cell_1/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!while/lstm_cell_1/dropout_4/Const?
while/lstm_cell_1/dropout_4/MulMul&while/lstm_cell_1/ones_like_1:output:0*while/lstm_cell_1/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell_1/dropout_4/Mul?
!while/lstm_cell_1/dropout_4/ShapeShape&while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_4/Shape?
8while/lstm_cell_1/dropout_4/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_1/dropout_4/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_1/dropout_4/GreaterEqual/y?
(while/lstm_cell_1/dropout_4/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_4/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2*
(while/lstm_cell_1/dropout_4/GreaterEqual?
 while/lstm_cell_1/dropout_4/CastCast,while/lstm_cell_1/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2"
 while/lstm_cell_1/dropout_4/Cast?
!while/lstm_cell_1/dropout_4/Mul_1Mul#while/lstm_cell_1/dropout_4/Mul:z:0$while/lstm_cell_1/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2#
!while/lstm_cell_1/dropout_4/Mul_1?
!while/lstm_cell_1/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!while/lstm_cell_1/dropout_5/Const?
while/lstm_cell_1/dropout_5/MulMul&while/lstm_cell_1/ones_like_1:output:0*while/lstm_cell_1/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell_1/dropout_5/Mul?
!while/lstm_cell_1/dropout_5/ShapeShape&while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_5/Shape?
8while/lstm_cell_1/dropout_5/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_1/dropout_5/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_1/dropout_5/GreaterEqual/y?
(while/lstm_cell_1/dropout_5/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_5/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2*
(while/lstm_cell_1/dropout_5/GreaterEqual?
 while/lstm_cell_1/dropout_5/CastCast,while/lstm_cell_1/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2"
 while/lstm_cell_1/dropout_5/Cast?
!while/lstm_cell_1/dropout_5/Mul_1Mul#while/lstm_cell_1/dropout_5/Mul:z:0$while/lstm_cell_1/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2#
!while/lstm_cell_1/dropout_5/Mul_1?
!while/lstm_cell_1/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!while/lstm_cell_1/dropout_6/Const?
while/lstm_cell_1/dropout_6/MulMul&while/lstm_cell_1/ones_like_1:output:0*while/lstm_cell_1/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell_1/dropout_6/Mul?
!while/lstm_cell_1/dropout_6/ShapeShape&while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_6/Shape?
8while/lstm_cell_1/dropout_6/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_1/dropout_6/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_1/dropout_6/GreaterEqual/y?
(while/lstm_cell_1/dropout_6/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_6/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2*
(while/lstm_cell_1/dropout_6/GreaterEqual?
 while/lstm_cell_1/dropout_6/CastCast,while/lstm_cell_1/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2"
 while/lstm_cell_1/dropout_6/Cast?
!while/lstm_cell_1/dropout_6/Mul_1Mul#while/lstm_cell_1/dropout_6/Mul:z:0$while/lstm_cell_1/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2#
!while/lstm_cell_1/dropout_6/Mul_1?
!while/lstm_cell_1/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!while/lstm_cell_1/dropout_7/Const?
while/lstm_cell_1/dropout_7/MulMul&while/lstm_cell_1/ones_like_1:output:0*while/lstm_cell_1/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell_1/dropout_7/Mul?
!while/lstm_cell_1/dropout_7/ShapeShape&while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_7/Shape?
8while/lstm_cell_1/dropout_7/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed22:
8while/lstm_cell_1/dropout_7/random_uniform/RandomUniform?
*while/lstm_cell_1/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_1/dropout_7/GreaterEqual/y?
(while/lstm_cell_1/dropout_7/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_7/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2*
(while/lstm_cell_1/dropout_7/GreaterEqual?
 while/lstm_cell_1/dropout_7/CastCast,while/lstm_cell_1/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2"
 while/lstm_cell_1/dropout_7/Cast?
!while/lstm_cell_1/dropout_7/Mul_1Mul#while/lstm_cell_1/dropout_7/Mul:z:0$while/lstm_cell_1/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2#
!while/lstm_cell_1/dropout_7/Mul_1?
while/lstm_cell_1/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/mul?
while/lstm_cell_1/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/mul_1?
while/lstm_cell_1/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/mul_2?
while/lstm_cell_1/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2
while/lstm_cell_1/mul_3t
while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const?
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_1/split/split_dim?
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes
:	P? *
dtype02(
&while/lstm_cell_1/split/ReadVariableOp?
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	P?:	P?:	P?:	P?*
	num_split2
while/lstm_cell_1/split?
while/lstm_cell_1/MatMulMatMulwhile/lstm_cell_1/mul:z:0 while/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul?
while/lstm_cell_1/MatMul_1MatMulwhile/lstm_cell_1/mul_1:z:0 while/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_1?
while/lstm_cell_1/MatMul_2MatMulwhile/lstm_cell_1/mul_2:z:0 while/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_2?
while/lstm_cell_1/MatMul_3MatMulwhile/lstm_cell_1/mul_3:z:0 while/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_3x
while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const_1?
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_1/split_1/split_dim?
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes	
:? *
dtype02*
(while/lstm_cell_1/split_1/ReadVariableOp?
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
while/lstm_cell_1/split_1?
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/BiasAdd?
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/BiasAdd_1?
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/BiasAdd_2?
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/BiasAdd_3?
while/lstm_cell_1/mul_4Mulwhile_placeholder_2%while/lstm_cell_1/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_4?
while/lstm_cell_1/mul_5Mulwhile_placeholder_2%while/lstm_cell_1/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_5?
while/lstm_cell_1/mul_6Mulwhile_placeholder_2%while/lstm_cell_1/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_6?
while/lstm_cell_1/mul_7Mulwhile_placeholder_2%while/lstm_cell_1/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_7?
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02"
 while/lstm_cell_1/ReadVariableOp?
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_1/strided_slice/stack?
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice/stack_1?
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_1/strided_slice/stack_2?
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell_1/strided_slice?
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul_4:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_4?
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add?
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Sigmoid?
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02$
"while/lstm_cell_1/ReadVariableOp_1?
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_1/stack?
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/lstm_cell_1/strided_slice_1/stack_1?
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_1/stack_2?
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_1?
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_5:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_5?
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add_1?
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Sigmoid_1?
while/lstm_cell_1/mul_8Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_8?
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02$
"while/lstm_cell_1/ReadVariableOp_2?
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_2/stack?
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/lstm_cell_1/strided_slice_2/stack_1?
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_2/stack_2?
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_2?
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_6:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_6?
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add_2?
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Tanh?
while/lstm_cell_1/mul_9Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_9?
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_8:z:0while/lstm_cell_1/mul_9:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add_3?
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
?? *
dtype02$
"while/lstm_cell_1/ReadVariableOp_3?
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_3/stack?
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_1/strided_slice_3/stack_1?
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_3/stack_2?
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_3?
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_7:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/MatMul_7?
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/add_4?
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Sigmoid_2?
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/Tanh_1?
while/lstm_cell_1/mul_10Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_1/mul_10?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_1/mul_10:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_1/add_3:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?
B__inference_layer0_layer_call_and_return_conditional_losses_348809

inputs-
)lstm_cell_1_split_readvariableop_resource/
+lstm_cell_1_split_1_readvariableop_resource'
#lstm_cell_1_readvariableop_resource
identity??lstm_cell_1/ReadVariableOp?lstm_cell_1/ReadVariableOp_1?lstm_cell_1/ReadVariableOp_2?lstm_cell_1/ReadVariableOp_3? lstm_cell_1/split/ReadVariableOp?"lstm_cell_1/split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????P2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask2
strided_slice_2?
lstm_cell_1/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like/Shape
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_1/ones_like/Const?
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/ones_like|
lstm_cell_1/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like_1/Shape?
lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_1/ones_like_1/Const?
lstm_cell_1/ones_like_1Fill&lstm_cell_1/ones_like_1/Shape:output:0&lstm_cell_1/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/ones_like_1?
lstm_cell_1/mulMulstrided_slice_2:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/mul?
lstm_cell_1/mul_1Mulstrided_slice_2:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/mul_1?
lstm_cell_1/mul_2Mulstrided_slice_2:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/mul_2?
lstm_cell_1/mul_3Mulstrided_slice_2:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2
lstm_cell_1/mul_3h
lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dim?
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource*
_output_shapes
:	P? *
dtype02"
 lstm_cell_1/split/ReadVariableOp?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	P?:	P?:	P?:	P?*
	num_split2
lstm_cell_1/split?
lstm_cell_1/MatMulMatMullstm_cell_1/mul:z:0lstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul?
lstm_cell_1/MatMul_1MatMullstm_cell_1/mul_1:z:0lstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_1?
lstm_cell_1/MatMul_2MatMullstm_cell_1/mul_2:z:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_2?
lstm_cell_1/MatMul_3MatMullstm_cell_1/mul_3:z:0lstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_3l
lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const_1?
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_1/split_1/split_dim?
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes	
:? *
dtype02$
"lstm_cell_1/split_1/ReadVariableOp?
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell_1/split_1?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/BiasAdd?
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_1/BiasAdd_1?
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_1/BiasAdd_2?
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_1/BiasAdd_3?
lstm_cell_1/mul_4Mulzeros:output:0 lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_4?
lstm_cell_1/mul_5Mulzeros:output:0 lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_5?
lstm_cell_1/mul_6Mulzeros:output:0 lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_6?
lstm_cell_1/mul_7Mulzeros:output:0 lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_7?
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
lstm_cell_1/ReadVariableOp?
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_1/strided_slice/stack?
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice/stack_1?
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_1/strided_slice/stack_2?
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_1/strided_slice?
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul_4:z:0"lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_4?
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add}
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Sigmoid?
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
lstm_cell_1/ReadVariableOp_1?
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_1/stack?
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#lstm_cell_1/strided_slice_1/stack_1?
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_1/stack_2?
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_1?
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_5:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_5?
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add_1?
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Sigmoid_1?
lstm_cell_1/mul_8Mullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_8?
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
lstm_cell_1/ReadVariableOp_2?
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_2/stack?
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#lstm_cell_1/strided_slice_2/stack_1?
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_2/stack_2?
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_2?
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_6:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_6?
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add_2v
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Tanh?
lstm_cell_1/mul_9Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_9?
lstm_cell_1/add_3AddV2lstm_cell_1/mul_8:z:0lstm_cell_1/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add_3?
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
lstm_cell_1/ReadVariableOp_3?
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_3/stack?
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_1/strided_slice_3/stack_1?
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_3/stack_2?
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_3?
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_7:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/MatMul_7?
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/add_4?
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Sigmoid_2z
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/Tanh_1?
lstm_cell_1/mul_10Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_1/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_348673*
condR
while_cond_348672*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????P:::28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
??
?
!__inference__wrapped_model_347392
layer0_inputB
>sequential_12_layer0_lstm_cell_1_split_readvariableop_resourceD
@sequential_12_layer0_lstm_cell_1_split_1_readvariableop_resource<
8sequential_12_layer0_lstm_cell_1_readvariableop_resource9
5sequential_12_dense_10_matmul_readvariableop_resource:
6sequential_12_dense_10_biasadd_readvariableop_resource7
3sequential_12_output_matmul_readvariableop_resource8
4sequential_12_output_biasadd_readvariableop_resource
identity??-sequential_12/dense_10/BiasAdd/ReadVariableOp?,sequential_12/dense_10/MatMul/ReadVariableOp?/sequential_12/layer0/lstm_cell_1/ReadVariableOp?1sequential_12/layer0/lstm_cell_1/ReadVariableOp_1?1sequential_12/layer0/lstm_cell_1/ReadVariableOp_2?1sequential_12/layer0/lstm_cell_1/ReadVariableOp_3?5sequential_12/layer0/lstm_cell_1/split/ReadVariableOp?7sequential_12/layer0/lstm_cell_1/split_1/ReadVariableOp?sequential_12/layer0/while?+sequential_12/output/BiasAdd/ReadVariableOp?*sequential_12/output/MatMul/ReadVariableOpt
sequential_12/layer0/ShapeShapelayer0_input*
T0*
_output_shapes
:2
sequential_12/layer0/Shape?
(sequential_12/layer0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_12/layer0/strided_slice/stack?
*sequential_12/layer0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_12/layer0/strided_slice/stack_1?
*sequential_12/layer0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_12/layer0/strided_slice/stack_2?
"sequential_12/layer0/strided_sliceStridedSlice#sequential_12/layer0/Shape:output:01sequential_12/layer0/strided_slice/stack:output:03sequential_12/layer0/strided_slice/stack_1:output:03sequential_12/layer0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_12/layer0/strided_slice?
 sequential_12/layer0/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 sequential_12/layer0/zeros/mul/y?
sequential_12/layer0/zeros/mulMul+sequential_12/layer0/strided_slice:output:0)sequential_12/layer0/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_12/layer0/zeros/mul?
!sequential_12/layer0/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!sequential_12/layer0/zeros/Less/y?
sequential_12/layer0/zeros/LessLess"sequential_12/layer0/zeros/mul:z:0*sequential_12/layer0/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_12/layer0/zeros/Less?
#sequential_12/layer0/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2%
#sequential_12/layer0/zeros/packed/1?
!sequential_12/layer0/zeros/packedPack+sequential_12/layer0/strided_slice:output:0,sequential_12/layer0/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_12/layer0/zeros/packed?
 sequential_12/layer0/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_12/layer0/zeros/Const?
sequential_12/layer0/zerosFill*sequential_12/layer0/zeros/packed:output:0)sequential_12/layer0/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_12/layer0/zeros?
"sequential_12/layer0/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential_12/layer0/zeros_1/mul/y?
 sequential_12/layer0/zeros_1/mulMul+sequential_12/layer0/strided_slice:output:0+sequential_12/layer0/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_12/layer0/zeros_1/mul?
#sequential_12/layer0/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2%
#sequential_12/layer0/zeros_1/Less/y?
!sequential_12/layer0/zeros_1/LessLess$sequential_12/layer0/zeros_1/mul:z:0,sequential_12/layer0/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_12/layer0/zeros_1/Less?
%sequential_12/layer0/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2'
%sequential_12/layer0/zeros_1/packed/1?
#sequential_12/layer0/zeros_1/packedPack+sequential_12/layer0/strided_slice:output:0.sequential_12/layer0/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_12/layer0/zeros_1/packed?
"sequential_12/layer0/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_12/layer0/zeros_1/Const?
sequential_12/layer0/zeros_1Fill,sequential_12/layer0/zeros_1/packed:output:0+sequential_12/layer0/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_12/layer0/zeros_1?
#sequential_12/layer0/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_12/layer0/transpose/perm?
sequential_12/layer0/transpose	Transposelayer0_input,sequential_12/layer0/transpose/perm:output:0*
T0*+
_output_shapes
:?????????P2 
sequential_12/layer0/transpose?
sequential_12/layer0/Shape_1Shape"sequential_12/layer0/transpose:y:0*
T0*
_output_shapes
:2
sequential_12/layer0/Shape_1?
*sequential_12/layer0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_12/layer0/strided_slice_1/stack?
,sequential_12/layer0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_12/layer0/strided_slice_1/stack_1?
,sequential_12/layer0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_12/layer0/strided_slice_1/stack_2?
$sequential_12/layer0/strided_slice_1StridedSlice%sequential_12/layer0/Shape_1:output:03sequential_12/layer0/strided_slice_1/stack:output:05sequential_12/layer0/strided_slice_1/stack_1:output:05sequential_12/layer0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_12/layer0/strided_slice_1?
0sequential_12/layer0/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0sequential_12/layer0/TensorArrayV2/element_shape?
"sequential_12/layer0/TensorArrayV2TensorListReserve9sequential_12/layer0/TensorArrayV2/element_shape:output:0-sequential_12/layer0/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_12/layer0/TensorArrayV2?
Jsequential_12/layer0/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   2L
Jsequential_12/layer0/TensorArrayUnstack/TensorListFromTensor/element_shape?
<sequential_12/layer0/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_12/layer0/transpose:y:0Ssequential_12/layer0/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_12/layer0/TensorArrayUnstack/TensorListFromTensor?
*sequential_12/layer0/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_12/layer0/strided_slice_2/stack?
,sequential_12/layer0/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_12/layer0/strided_slice_2/stack_1?
,sequential_12/layer0/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_12/layer0/strided_slice_2/stack_2?
$sequential_12/layer0/strided_slice_2StridedSlice"sequential_12/layer0/transpose:y:03sequential_12/layer0/strided_slice_2/stack:output:05sequential_12/layer0/strided_slice_2/stack_1:output:05sequential_12/layer0/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask2&
$sequential_12/layer0/strided_slice_2?
0sequential_12/layer0/lstm_cell_1/ones_like/ShapeShape-sequential_12/layer0/strided_slice_2:output:0*
T0*
_output_shapes
:22
0sequential_12/layer0/lstm_cell_1/ones_like/Shape?
0sequential_12/layer0/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??22
0sequential_12/layer0/lstm_cell_1/ones_like/Const?
*sequential_12/layer0/lstm_cell_1/ones_likeFill9sequential_12/layer0/lstm_cell_1/ones_like/Shape:output:09sequential_12/layer0/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????P2,
*sequential_12/layer0/lstm_cell_1/ones_like?
2sequential_12/layer0/lstm_cell_1/ones_like_1/ShapeShape#sequential_12/layer0/zeros:output:0*
T0*
_output_shapes
:24
2sequential_12/layer0/lstm_cell_1/ones_like_1/Shape?
2sequential_12/layer0/lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2sequential_12/layer0/lstm_cell_1/ones_like_1/Const?
,sequential_12/layer0/lstm_cell_1/ones_like_1Fill;sequential_12/layer0/lstm_cell_1/ones_like_1/Shape:output:0;sequential_12/layer0/lstm_cell_1/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2.
,sequential_12/layer0/lstm_cell_1/ones_like_1?
$sequential_12/layer0/lstm_cell_1/mulMul-sequential_12/layer0/strided_slice_2:output:03sequential_12/layer0/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2&
$sequential_12/layer0/lstm_cell_1/mul?
&sequential_12/layer0/lstm_cell_1/mul_1Mul-sequential_12/layer0/strided_slice_2:output:03sequential_12/layer0/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2(
&sequential_12/layer0/lstm_cell_1/mul_1?
&sequential_12/layer0/lstm_cell_1/mul_2Mul-sequential_12/layer0/strided_slice_2:output:03sequential_12/layer0/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2(
&sequential_12/layer0/lstm_cell_1/mul_2?
&sequential_12/layer0/lstm_cell_1/mul_3Mul-sequential_12/layer0/strided_slice_2:output:03sequential_12/layer0/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:?????????P2(
&sequential_12/layer0/lstm_cell_1/mul_3?
&sequential_12/layer0/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_12/layer0/lstm_cell_1/Const?
0sequential_12/layer0/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential_12/layer0/lstm_cell_1/split/split_dim?
5sequential_12/layer0/lstm_cell_1/split/ReadVariableOpReadVariableOp>sequential_12_layer0_lstm_cell_1_split_readvariableop_resource*
_output_shapes
:	P? *
dtype027
5sequential_12/layer0/lstm_cell_1/split/ReadVariableOp?
&sequential_12/layer0/lstm_cell_1/splitSplit9sequential_12/layer0/lstm_cell_1/split/split_dim:output:0=sequential_12/layer0/lstm_cell_1/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	P?:	P?:	P?:	P?*
	num_split2(
&sequential_12/layer0/lstm_cell_1/split?
'sequential_12/layer0/lstm_cell_1/MatMulMatMul(sequential_12/layer0/lstm_cell_1/mul:z:0/sequential_12/layer0/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:??????????2)
'sequential_12/layer0/lstm_cell_1/MatMul?
)sequential_12/layer0/lstm_cell_1/MatMul_1MatMul*sequential_12/layer0/lstm_cell_1/mul_1:z:0/sequential_12/layer0/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:??????????2+
)sequential_12/layer0/lstm_cell_1/MatMul_1?
)sequential_12/layer0/lstm_cell_1/MatMul_2MatMul*sequential_12/layer0/lstm_cell_1/mul_2:z:0/sequential_12/layer0/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:??????????2+
)sequential_12/layer0/lstm_cell_1/MatMul_2?
)sequential_12/layer0/lstm_cell_1/MatMul_3MatMul*sequential_12/layer0/lstm_cell_1/mul_3:z:0/sequential_12/layer0/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:??????????2+
)sequential_12/layer0/lstm_cell_1/MatMul_3?
(sequential_12/layer0/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_12/layer0/lstm_cell_1/Const_1?
2sequential_12/layer0/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 24
2sequential_12/layer0/lstm_cell_1/split_1/split_dim?
7sequential_12/layer0/lstm_cell_1/split_1/ReadVariableOpReadVariableOp@sequential_12_layer0_lstm_cell_1_split_1_readvariableop_resource*
_output_shapes	
:? *
dtype029
7sequential_12/layer0/lstm_cell_1/split_1/ReadVariableOp?
(sequential_12/layer0/lstm_cell_1/split_1Split;sequential_12/layer0/lstm_cell_1/split_1/split_dim:output:0?sequential_12/layer0/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2*
(sequential_12/layer0/lstm_cell_1/split_1?
(sequential_12/layer0/lstm_cell_1/BiasAddBiasAdd1sequential_12/layer0/lstm_cell_1/MatMul:product:01sequential_12/layer0/lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:??????????2*
(sequential_12/layer0/lstm_cell_1/BiasAdd?
*sequential_12/layer0/lstm_cell_1/BiasAdd_1BiasAdd3sequential_12/layer0/lstm_cell_1/MatMul_1:product:01sequential_12/layer0/lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:??????????2,
*sequential_12/layer0/lstm_cell_1/BiasAdd_1?
*sequential_12/layer0/lstm_cell_1/BiasAdd_2BiasAdd3sequential_12/layer0/lstm_cell_1/MatMul_2:product:01sequential_12/layer0/lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:??????????2,
*sequential_12/layer0/lstm_cell_1/BiasAdd_2?
*sequential_12/layer0/lstm_cell_1/BiasAdd_3BiasAdd3sequential_12/layer0/lstm_cell_1/MatMul_3:product:01sequential_12/layer0/lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:??????????2,
*sequential_12/layer0/lstm_cell_1/BiasAdd_3?
&sequential_12/layer0/lstm_cell_1/mul_4Mul#sequential_12/layer0/zeros:output:05sequential_12/layer0/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2(
&sequential_12/layer0/lstm_cell_1/mul_4?
&sequential_12/layer0/lstm_cell_1/mul_5Mul#sequential_12/layer0/zeros:output:05sequential_12/layer0/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2(
&sequential_12/layer0/lstm_cell_1/mul_5?
&sequential_12/layer0/lstm_cell_1/mul_6Mul#sequential_12/layer0/zeros:output:05sequential_12/layer0/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2(
&sequential_12/layer0/lstm_cell_1/mul_6?
&sequential_12/layer0/lstm_cell_1/mul_7Mul#sequential_12/layer0/zeros:output:05sequential_12/layer0/lstm_cell_1/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2(
&sequential_12/layer0/lstm_cell_1/mul_7?
/sequential_12/layer0/lstm_cell_1/ReadVariableOpReadVariableOp8sequential_12_layer0_lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype021
/sequential_12/layer0/lstm_cell_1/ReadVariableOp?
4sequential_12/layer0/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        26
4sequential_12/layer0/lstm_cell_1/strided_slice/stack?
6sequential_12/layer0/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       28
6sequential_12/layer0/lstm_cell_1/strided_slice/stack_1?
6sequential_12/layer0/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6sequential_12/layer0/lstm_cell_1/strided_slice/stack_2?
.sequential_12/layer0/lstm_cell_1/strided_sliceStridedSlice7sequential_12/layer0/lstm_cell_1/ReadVariableOp:value:0=sequential_12/layer0/lstm_cell_1/strided_slice/stack:output:0?sequential_12/layer0/lstm_cell_1/strided_slice/stack_1:output:0?sequential_12/layer0/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask20
.sequential_12/layer0/lstm_cell_1/strided_slice?
)sequential_12/layer0/lstm_cell_1/MatMul_4MatMul*sequential_12/layer0/lstm_cell_1/mul_4:z:07sequential_12/layer0/lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:??????????2+
)sequential_12/layer0/lstm_cell_1/MatMul_4?
$sequential_12/layer0/lstm_cell_1/addAddV21sequential_12/layer0/lstm_cell_1/BiasAdd:output:03sequential_12/layer0/lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2&
$sequential_12/layer0/lstm_cell_1/add?
(sequential_12/layer0/lstm_cell_1/SigmoidSigmoid(sequential_12/layer0/lstm_cell_1/add:z:0*
T0*(
_output_shapes
:??????????2*
(sequential_12/layer0/lstm_cell_1/Sigmoid?
1sequential_12/layer0/lstm_cell_1/ReadVariableOp_1ReadVariableOp8sequential_12_layer0_lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype023
1sequential_12/layer0/lstm_cell_1/ReadVariableOp_1?
6sequential_12/layer0/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       28
6sequential_12/layer0/lstm_cell_1/strided_slice_1/stack?
8sequential_12/layer0/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2:
8sequential_12/layer0/lstm_cell_1/strided_slice_1/stack_1?
8sequential_12/layer0/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential_12/layer0/lstm_cell_1/strided_slice_1/stack_2?
0sequential_12/layer0/lstm_cell_1/strided_slice_1StridedSlice9sequential_12/layer0/lstm_cell_1/ReadVariableOp_1:value:0?sequential_12/layer0/lstm_cell_1/strided_slice_1/stack:output:0Asequential_12/layer0/lstm_cell_1/strided_slice_1/stack_1:output:0Asequential_12/layer0/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask22
0sequential_12/layer0/lstm_cell_1/strided_slice_1?
)sequential_12/layer0/lstm_cell_1/MatMul_5MatMul*sequential_12/layer0/lstm_cell_1/mul_5:z:09sequential_12/layer0/lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2+
)sequential_12/layer0/lstm_cell_1/MatMul_5?
&sequential_12/layer0/lstm_cell_1/add_1AddV23sequential_12/layer0/lstm_cell_1/BiasAdd_1:output:03sequential_12/layer0/lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2(
&sequential_12/layer0/lstm_cell_1/add_1?
*sequential_12/layer0/lstm_cell_1/Sigmoid_1Sigmoid*sequential_12/layer0/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:??????????2,
*sequential_12/layer0/lstm_cell_1/Sigmoid_1?
&sequential_12/layer0/lstm_cell_1/mul_8Mul.sequential_12/layer0/lstm_cell_1/Sigmoid_1:y:0%sequential_12/layer0/zeros_1:output:0*
T0*(
_output_shapes
:??????????2(
&sequential_12/layer0/lstm_cell_1/mul_8?
1sequential_12/layer0/lstm_cell_1/ReadVariableOp_2ReadVariableOp8sequential_12_layer0_lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype023
1sequential_12/layer0/lstm_cell_1/ReadVariableOp_2?
6sequential_12/layer0/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       28
6sequential_12/layer0/lstm_cell_1/strided_slice_2/stack?
8sequential_12/layer0/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2:
8sequential_12/layer0/lstm_cell_1/strided_slice_2/stack_1?
8sequential_12/layer0/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential_12/layer0/lstm_cell_1/strided_slice_2/stack_2?
0sequential_12/layer0/lstm_cell_1/strided_slice_2StridedSlice9sequential_12/layer0/lstm_cell_1/ReadVariableOp_2:value:0?sequential_12/layer0/lstm_cell_1/strided_slice_2/stack:output:0Asequential_12/layer0/lstm_cell_1/strided_slice_2/stack_1:output:0Asequential_12/layer0/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask22
0sequential_12/layer0/lstm_cell_1/strided_slice_2?
)sequential_12/layer0/lstm_cell_1/MatMul_6MatMul*sequential_12/layer0/lstm_cell_1/mul_6:z:09sequential_12/layer0/lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2+
)sequential_12/layer0/lstm_cell_1/MatMul_6?
&sequential_12/layer0/lstm_cell_1/add_2AddV23sequential_12/layer0/lstm_cell_1/BiasAdd_2:output:03sequential_12/layer0/lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2(
&sequential_12/layer0/lstm_cell_1/add_2?
%sequential_12/layer0/lstm_cell_1/TanhTanh*sequential_12/layer0/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:??????????2'
%sequential_12/layer0/lstm_cell_1/Tanh?
&sequential_12/layer0/lstm_cell_1/mul_9Mul,sequential_12/layer0/lstm_cell_1/Sigmoid:y:0)sequential_12/layer0/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:??????????2(
&sequential_12/layer0/lstm_cell_1/mul_9?
&sequential_12/layer0/lstm_cell_1/add_3AddV2*sequential_12/layer0/lstm_cell_1/mul_8:z:0*sequential_12/layer0/lstm_cell_1/mul_9:z:0*
T0*(
_output_shapes
:??????????2(
&sequential_12/layer0/lstm_cell_1/add_3?
1sequential_12/layer0/lstm_cell_1/ReadVariableOp_3ReadVariableOp8sequential_12_layer0_lstm_cell_1_readvariableop_resource* 
_output_shapes
:
?? *
dtype023
1sequential_12/layer0/lstm_cell_1/ReadVariableOp_3?
6sequential_12/layer0/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       28
6sequential_12/layer0/lstm_cell_1/strided_slice_3/stack?
8sequential_12/layer0/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2:
8sequential_12/layer0/lstm_cell_1/strided_slice_3/stack_1?
8sequential_12/layer0/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential_12/layer0/lstm_cell_1/strided_slice_3/stack_2?
0sequential_12/layer0/lstm_cell_1/strided_slice_3StridedSlice9sequential_12/layer0/lstm_cell_1/ReadVariableOp_3:value:0?sequential_12/layer0/lstm_cell_1/strided_slice_3/stack:output:0Asequential_12/layer0/lstm_cell_1/strided_slice_3/stack_1:output:0Asequential_12/layer0/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask22
0sequential_12/layer0/lstm_cell_1/strided_slice_3?
)sequential_12/layer0/lstm_cell_1/MatMul_7MatMul*sequential_12/layer0/lstm_cell_1/mul_7:z:09sequential_12/layer0/lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2+
)sequential_12/layer0/lstm_cell_1/MatMul_7?
&sequential_12/layer0/lstm_cell_1/add_4AddV23sequential_12/layer0/lstm_cell_1/BiasAdd_3:output:03sequential_12/layer0/lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2(
&sequential_12/layer0/lstm_cell_1/add_4?
*sequential_12/layer0/lstm_cell_1/Sigmoid_2Sigmoid*sequential_12/layer0/lstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:??????????2,
*sequential_12/layer0/lstm_cell_1/Sigmoid_2?
'sequential_12/layer0/lstm_cell_1/Tanh_1Tanh*sequential_12/layer0/lstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:??????????2)
'sequential_12/layer0/lstm_cell_1/Tanh_1?
'sequential_12/layer0/lstm_cell_1/mul_10Mul.sequential_12/layer0/lstm_cell_1/Sigmoid_2:y:0+sequential_12/layer0/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2)
'sequential_12/layer0/lstm_cell_1/mul_10?
2sequential_12/layer0/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   24
2sequential_12/layer0/TensorArrayV2_1/element_shape?
$sequential_12/layer0/TensorArrayV2_1TensorListReserve;sequential_12/layer0/TensorArrayV2_1/element_shape:output:0-sequential_12/layer0/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_12/layer0/TensorArrayV2_1x
sequential_12/layer0/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_12/layer0/time?
-sequential_12/layer0/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-sequential_12/layer0/while/maximum_iterations?
'sequential_12/layer0/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_12/layer0/while/loop_counter?
sequential_12/layer0/whileWhile0sequential_12/layer0/while/loop_counter:output:06sequential_12/layer0/while/maximum_iterations:output:0"sequential_12/layer0/time:output:0-sequential_12/layer0/TensorArrayV2_1:handle:0#sequential_12/layer0/zeros:output:0%sequential_12/layer0/zeros_1:output:0-sequential_12/layer0/strided_slice_1:output:0Lsequential_12/layer0/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_12_layer0_lstm_cell_1_split_readvariableop_resource@sequential_12_layer0_lstm_cell_1_split_1_readvariableop_resource8sequential_12_layer0_lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*2
body*R(
&sequential_12_layer0_while_body_347242*2
cond*R(
&sequential_12_layer0_while_cond_347241*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
sequential_12/layer0/while?
Esequential_12/layer0/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2G
Esequential_12/layer0/TensorArrayV2Stack/TensorListStack/element_shape?
7sequential_12/layer0/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_12/layer0/while:output:3Nsequential_12/layer0/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype029
7sequential_12/layer0/TensorArrayV2Stack/TensorListStack?
*sequential_12/layer0/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2,
*sequential_12/layer0/strided_slice_3/stack?
,sequential_12/layer0/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_12/layer0/strided_slice_3/stack_1?
,sequential_12/layer0/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_12/layer0/strided_slice_3/stack_2?
$sequential_12/layer0/strided_slice_3StridedSlice@sequential_12/layer0/TensorArrayV2Stack/TensorListStack:tensor:03sequential_12/layer0/strided_slice_3/stack:output:05sequential_12/layer0/strided_slice_3/stack_1:output:05sequential_12/layer0/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2&
$sequential_12/layer0/strided_slice_3?
%sequential_12/layer0/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_12/layer0/transpose_1/perm?
 sequential_12/layer0/transpose_1	Transpose@sequential_12/layer0/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_12/layer0/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2"
 sequential_12/layer0/transpose_1?
sequential_12/layer0/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_12/layer0/runtime?
,sequential_12/dense_10/MatMul/ReadVariableOpReadVariableOp5sequential_12_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,sequential_12/dense_10/MatMul/ReadVariableOp?
sequential_12/dense_10/MatMulMatMul-sequential_12/layer0/strided_slice_3:output:04sequential_12/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_12/dense_10/MatMul?
-sequential_12/dense_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_12/dense_10/BiasAdd/ReadVariableOp?
sequential_12/dense_10/BiasAddBiasAdd'sequential_12/dense_10/MatMul:product:05sequential_12/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_12/dense_10/BiasAdd?
sequential_12/dense_10/ReluRelu'sequential_12/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_12/dense_10/Relu?
*sequential_12/output/MatMul/ReadVariableOpReadVariableOp3sequential_12_output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_12/output/MatMul/ReadVariableOp?
sequential_12/output/MatMulMatMul)sequential_12/dense_10/Relu:activations:02sequential_12/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_12/output/MatMul?
+sequential_12/output/BiasAdd/ReadVariableOpReadVariableOp4sequential_12_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_12/output/BiasAdd/ReadVariableOp?
sequential_12/output/BiasAddBiasAdd%sequential_12/output/MatMul:product:03sequential_12/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_12/output/BiasAdd?
sequential_12/output/SoftmaxSoftmax%sequential_12/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_12/output/Softmax?
IdentityIdentity&sequential_12/output/Softmax:softmax:0.^sequential_12/dense_10/BiasAdd/ReadVariableOp-^sequential_12/dense_10/MatMul/ReadVariableOp0^sequential_12/layer0/lstm_cell_1/ReadVariableOp2^sequential_12/layer0/lstm_cell_1/ReadVariableOp_12^sequential_12/layer0/lstm_cell_1/ReadVariableOp_22^sequential_12/layer0/lstm_cell_1/ReadVariableOp_36^sequential_12/layer0/lstm_cell_1/split/ReadVariableOp8^sequential_12/layer0/lstm_cell_1/split_1/ReadVariableOp^sequential_12/layer0/while,^sequential_12/output/BiasAdd/ReadVariableOp+^sequential_12/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????P:::::::2^
-sequential_12/dense_10/BiasAdd/ReadVariableOp-sequential_12/dense_10/BiasAdd/ReadVariableOp2\
,sequential_12/dense_10/MatMul/ReadVariableOp,sequential_12/dense_10/MatMul/ReadVariableOp2b
/sequential_12/layer0/lstm_cell_1/ReadVariableOp/sequential_12/layer0/lstm_cell_1/ReadVariableOp2f
1sequential_12/layer0/lstm_cell_1/ReadVariableOp_11sequential_12/layer0/lstm_cell_1/ReadVariableOp_12f
1sequential_12/layer0/lstm_cell_1/ReadVariableOp_21sequential_12/layer0/lstm_cell_1/ReadVariableOp_22f
1sequential_12/layer0/lstm_cell_1/ReadVariableOp_31sequential_12/layer0/lstm_cell_1/ReadVariableOp_32n
5sequential_12/layer0/lstm_cell_1/split/ReadVariableOp5sequential_12/layer0/lstm_cell_1/split/ReadVariableOp2r
7sequential_12/layer0/lstm_cell_1/split_1/ReadVariableOp7sequential_12/layer0/lstm_cell_1/split_1/ReadVariableOp28
sequential_12/layer0/whilesequential_12/layer0/while2Z
+sequential_12/output/BiasAdd/ReadVariableOp+sequential_12/output/BiasAdd/ReadVariableOp2X
*sequential_12/output/MatMul/ReadVariableOp*sequential_12/output/MatMul/ReadVariableOp:Y U
+
_output_shapes
:?????????P
&
_user_specified_namelayer0_input
?
?
while_cond_347957
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_347957___redundant_placeholder04
0while_while_cond_347957___redundant_placeholder14
0while_while_cond_347957___redundant_placeholder24
0while_while_cond_347957___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?	
?
D__inference_dense_10_layer_call_and_return_conditional_losses_348850

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_layer0_layer_call_fn_351049

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer0_layer_call_and_return_conditional_losses_3488092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????P:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
&sequential_12_layer0_while_cond_347241F
Bsequential_12_layer0_while_sequential_12_layer0_while_loop_counterL
Hsequential_12_layer0_while_sequential_12_layer0_while_maximum_iterations*
&sequential_12_layer0_while_placeholder,
(sequential_12_layer0_while_placeholder_1,
(sequential_12_layer0_while_placeholder_2,
(sequential_12_layer0_while_placeholder_3H
Dsequential_12_layer0_while_less_sequential_12_layer0_strided_slice_1^
Zsequential_12_layer0_while_sequential_12_layer0_while_cond_347241___redundant_placeholder0^
Zsequential_12_layer0_while_sequential_12_layer0_while_cond_347241___redundant_placeholder1^
Zsequential_12_layer0_while_sequential_12_layer0_while_cond_347241___redundant_placeholder2^
Zsequential_12_layer0_while_sequential_12_layer0_while_cond_347241___redundant_placeholder3'
#sequential_12_layer0_while_identity
?
sequential_12/layer0/while/LessLess&sequential_12_layer0_while_placeholderDsequential_12_layer0_while_less_sequential_12_layer0_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_12/layer0/while/Less?
#sequential_12/layer0/while/IdentityIdentity#sequential_12/layer0/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_12/layer0/while/Identity"S
#sequential_12_layer0_while_identity,sequential_12/layer0/while/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
.__inference_sequential_12_layer_call_fn_349710

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_3489392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????P:::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
I
layer0_input9
serving_default_layer0_input:0?????????P:
output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?*
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
[__call__
\_default_save_signature
*]&call_and_return_all_conditional_losses"?'
_tf_keras_sequential?'{"class_name": "Sequential", "name": "sequential_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_12", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 80]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "layer0_input"}}, {"class_name": "LSTM", "config": {"name": "layer0", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 80]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 1024, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.3, "recurrent_dropout": 0.3, "implementation": 1}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 7, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 80]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 80]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_12", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 80]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "layer0_input"}}, {"class_name": "LSTM", "config": {"name": "layer0", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 80]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 1024, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.3, "recurrent_dropout": 0.3, "implementation": 1}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 7, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "categorical_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
^__call__
*_&call_and_return_all_conditional_losses"?
_tf_keras_rnn_layer?
{"class_name": "LSTM", "name": "layer0", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 80]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer0", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 80]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 1024, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.3, "recurrent_dropout": 0.3, "implementation": 1}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 80]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 80]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
`__call__
*a&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
b__call__
*c&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 7, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?
iter

beta_1

beta_2
	decay
 learning_ratemMmNmOmP!mQ"mR#mSvTvUvVvW!vX"vY#vZ"
	optimizer
Q
!0
"1
#2
3
4
5
6"
trackable_list_wrapper
Q
!0
"1
#2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
?
$layer_metrics
%non_trainable_variables

&layers
	variables
trainable_variables
'metrics
(layer_regularization_losses
regularization_losses
[__call__
\_default_save_signature
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
,
dserving_default"
signature_map
?

!kernel
"recurrent_kernel
#bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
e__call__
*f&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_1", "trainable": true, "dtype": "float32", "units": 1024, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.3, "recurrent_dropout": 0.3, "implementation": 1}}
 "
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
-layer_metrics
.non_trainable_variables

/layers
trainable_variables
	variables
0metrics

1states
2layer_regularization_losses
regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_10/kernel
:?2dense_10/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
3layer_metrics
4non_trainable_variables

5layers
trainable_variables
	variables
6metrics
7layer_regularization_losses
regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
 :	?2output/kernel
:2output/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
8layer_metrics
9non_trainable_variables

:layers
trainable_variables
	variables
;metrics
<layer_regularization_losses
regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
,:*	P? 2layer0/lstm_cell_1/kernel
7:5
?? 2#layer0/lstm_cell_1/recurrent_kernel
&:$? 2layer0/lstm_cell_1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
@non_trainable_variables

Alayers
)trainable_variables
*	variables
Bmetrics
Clayer_regularization_losses
+regularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'

0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	Dtotal
	Ecount
F	variables
G	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	Htotal
	Icount
J
_fn_kwargs
K	variables
L	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "categorical_accuracy", "dtype": "float32", "config": {"name": "categorical_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
D0
E1"
trackable_list_wrapper
-
F	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
H0
I1"
trackable_list_wrapper
-
K	variables"
_generic_user_object
(:&
??2Adam/dense_10/kernel/m
!:?2Adam/dense_10/bias/m
%:#	?2Adam/output/kernel/m
:2Adam/output/bias/m
1:/	P? 2 Adam/layer0/lstm_cell_1/kernel/m
<::
?? 2*Adam/layer0/lstm_cell_1/recurrent_kernel/m
+:)? 2Adam/layer0/lstm_cell_1/bias/m
(:&
??2Adam/dense_10/kernel/v
!:?2Adam/dense_10/bias/v
%:#	?2Adam/output/kernel/v
:2Adam/output/bias/v
1:/	P? 2 Adam/layer0/lstm_cell_1/kernel/v
<::
?? 2*Adam/layer0/lstm_cell_1/recurrent_kernel/v
+:)? 2Adam/layer0/lstm_cell_1/bias/v
?2?
.__inference_sequential_12_layer_call_fn_349710
.__inference_sequential_12_layer_call_fn_348956
.__inference_sequential_12_layer_call_fn_349729
.__inference_sequential_12_layer_call_fn_348996?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_347392?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? */?,
*?'
layer0_input?????????P
?2?
I__inference_sequential_12_layer_call_and_return_conditional_losses_348915
I__inference_sequential_12_layer_call_and_return_conditional_losses_349691
I__inference_sequential_12_layer_call_and_return_conditional_losses_348894
I__inference_sequential_12_layer_call_and_return_conditional_losses_349422?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_layer0_layer_call_fn_350389
'__inference_layer0_layer_call_fn_351049
'__inference_layer0_layer_call_fn_351038
'__inference_layer0_layer_call_fn_350378?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_layer0_layer_call_and_return_conditional_losses_350367
B__inference_layer0_layer_call_and_return_conditional_losses_350112
B__inference_layer0_layer_call_and_return_conditional_losses_350772
B__inference_layer0_layer_call_and_return_conditional_losses_351027?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_10_layer_call_fn_351069?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_10_layer_call_and_return_conditional_losses_351060?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_output_layer_call_fn_351089?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_output_layer_call_and_return_conditional_losses_351080?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_349025layer0_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_lstm_cell_1_layer_call_fn_351355
,__inference_lstm_cell_1_layer_call_fn_351338?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_351237
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_351321?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
!__inference__wrapped_model_347392u!#"9?6
/?,
*?'
layer0_input?????????P
? "/?,
*
output ?
output??????????
D__inference_dense_10_layer_call_and_return_conditional_losses_351060^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
)__inference_dense_10_layer_call_fn_351069Q0?-
&?#
!?
inputs??????????
? "????????????
B__inference_layer0_layer_call_and_return_conditional_losses_350112~!#"O?L
E?B
4?1
/?,
inputs/0??????????????????P

 
p

 
? "&?#
?
0??????????
? ?
B__inference_layer0_layer_call_and_return_conditional_losses_350367~!#"O?L
E?B
4?1
/?,
inputs/0??????????????????P

 
p 

 
? "&?#
?
0??????????
? ?
B__inference_layer0_layer_call_and_return_conditional_losses_350772n!#"??<
5?2
$?!
inputs?????????P

 
p

 
? "&?#
?
0??????????
? ?
B__inference_layer0_layer_call_and_return_conditional_losses_351027n!#"??<
5?2
$?!
inputs?????????P

 
p 

 
? "&?#
?
0??????????
? ?
'__inference_layer0_layer_call_fn_350378q!#"O?L
E?B
4?1
/?,
inputs/0??????????????????P

 
p

 
? "????????????
'__inference_layer0_layer_call_fn_350389q!#"O?L
E?B
4?1
/?,
inputs/0??????????????????P

 
p 

 
? "????????????
'__inference_layer0_layer_call_fn_351038a!#"??<
5?2
$?!
inputs?????????P

 
p

 
? "????????????
'__inference_layer0_layer_call_fn_351049a!#"??<
5?2
$?!
inputs?????????P

 
p 

 
? "????????????
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_351237?!#"??
x?u
 ?
inputs?????????P
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_351321?!#"??
x?u
 ?
inputs?????????P
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
,__inference_lstm_cell_1_layer_call_fn_351338?!#"??
x?u
 ?
inputs?????????P
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
,__inference_lstm_cell_1_layer_call_fn_351355?!#"??
x?u
 ?
inputs?????????P
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
B__inference_output_layer_call_and_return_conditional_losses_351080]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_output_layer_call_fn_351089P0?-
&?#
!?
inputs??????????
? "???????????
I__inference_sequential_12_layer_call_and_return_conditional_losses_348894s!#"A?>
7?4
*?'
layer0_input?????????P
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_12_layer_call_and_return_conditional_losses_348915s!#"A?>
7?4
*?'
layer0_input?????????P
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_12_layer_call_and_return_conditional_losses_349422m!#";?8
1?.
$?!
inputs?????????P
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_12_layer_call_and_return_conditional_losses_349691m!#";?8
1?.
$?!
inputs?????????P
p 

 
? "%?"
?
0?????????
? ?
.__inference_sequential_12_layer_call_fn_348956f!#"A?>
7?4
*?'
layer0_input?????????P
p

 
? "???????????
.__inference_sequential_12_layer_call_fn_348996f!#"A?>
7?4
*?'
layer0_input?????????P
p 

 
? "???????????
.__inference_sequential_12_layer_call_fn_349710`!#";?8
1?.
$?!
inputs?????????P
p

 
? "???????????
.__inference_sequential_12_layer_call_fn_349729`!#";?8
1?.
$?!
inputs?????????P
p 

 
? "???????????
$__inference_signature_wrapper_349025?!#"I?F
? 
??<
:
layer0_input*?'
layer0_input?????????P"/?,
*
output ?
output?????????