�	
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
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
.
Identity

input"T
output"T"	
Ttype
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
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
dtypetype�
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
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28��
�
batch_normalization_41/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_41/gamma
�
0batch_normalization_41/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_41/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_41/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_41/beta
�
/batch_normalization_41/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_41/beta*
_output_shapes	
:�*
dtype0
�
"batch_normalization_41/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_41/moving_mean
�
6batch_normalization_41/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_41/moving_mean*
_output_shapes	
:�*
dtype0
�
&batch_normalization_41/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_41/moving_variance
�
:batch_normalization_41/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_41/moving_variance*
_output_shapes	
:�*
dtype0
}
dense_164/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*!
shared_namedense_164/kernel
v
$dense_164/kernel/Read/ReadVariableOpReadVariableOpdense_164/kernel*
_output_shapes
:	�*
dtype0
t
dense_164/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_164/bias
m
"dense_164/bias/Read/ReadVariableOpReadVariableOpdense_164/bias*
_output_shapes
:*
dtype0
|
dense_165/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_165/kernel
u
$dense_165/kernel/Read/ReadVariableOpReadVariableOpdense_165/kernel*
_output_shapes

:*
dtype0
t
dense_165/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_165/bias
m
"dense_165/bias/Read/ReadVariableOpReadVariableOpdense_165/bias*
_output_shapes
:*
dtype0
|
dense_166/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_166/kernel
u
$dense_166/kernel/Read/ReadVariableOpReadVariableOpdense_166/kernel*
_output_shapes

:*
dtype0
t
dense_166/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_166/bias
m
"dense_166/bias/Read/ReadVariableOpReadVariableOpdense_166/bias*
_output_shapes
:*
dtype0
|
dense_167/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_167/kernel
u
$dense_167/kernel/Read/ReadVariableOpReadVariableOpdense_167/kernel*
_output_shapes

:*
dtype0
t
dense_167/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_167/bias
m
"dense_167/bias/Read/ReadVariableOpReadVariableOpdense_167/bias*
_output_shapes
:*
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
�
#Adam/batch_normalization_41/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/batch_normalization_41/gamma/m
�
7Adam/batch_normalization_41/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_41/gamma/m*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_41/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_41/beta/m
�
6Adam/batch_normalization_41/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_41/beta/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_164/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*(
shared_nameAdam/dense_164/kernel/m
�
+Adam/dense_164/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_164/kernel/m*
_output_shapes
:	�*
dtype0
�
Adam/dense_164/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_164/bias/m
{
)Adam/dense_164/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_164/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_165/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_165/kernel/m
�
+Adam/dense_165/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_165/kernel/m*
_output_shapes

:*
dtype0
�
Adam/dense_165/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_165/bias/m
{
)Adam/dense_165/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_165/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_166/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_166/kernel/m
�
+Adam/dense_166/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_166/kernel/m*
_output_shapes

:*
dtype0
�
Adam/dense_166/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_166/bias/m
{
)Adam/dense_166/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_166/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_167/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_167/kernel/m
�
+Adam/dense_167/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_167/kernel/m*
_output_shapes

:*
dtype0
�
Adam/dense_167/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_167/bias/m
{
)Adam/dense_167/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_167/bias/m*
_output_shapes
:*
dtype0
�
#Adam/batch_normalization_41/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/batch_normalization_41/gamma/v
�
7Adam/batch_normalization_41/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_41/gamma/v*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_41/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_41/beta/v
�
6Adam/batch_normalization_41/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_41/beta/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_164/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*(
shared_nameAdam/dense_164/kernel/v
�
+Adam/dense_164/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_164/kernel/v*
_output_shapes
:	�*
dtype0
�
Adam/dense_164/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_164/bias/v
{
)Adam/dense_164/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_164/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_165/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_165/kernel/v
�
+Adam/dense_165/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_165/kernel/v*
_output_shapes

:*
dtype0
�
Adam/dense_165/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_165/bias/v
{
)Adam/dense_165/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_165/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_166/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_166/kernel/v
�
+Adam/dense_166/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_166/kernel/v*
_output_shapes

:*
dtype0
�
Adam/dense_166/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_166/bias/v
{
)Adam/dense_166/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_166/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_167/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_167/kernel/v
�
+Adam/dense_167/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_167/kernel/v*
_output_shapes

:*
dtype0
�
Adam/dense_167/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_167/bias/v
{
)Adam/dense_167/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_167/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�8
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�7
value�7B�7 B�7
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
�
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
h

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
h

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
�
-iter

.beta_1

/beta_2
	0decay
1learning_ratemUmVmWmXmYmZ!m["m\'m](m^v_v`vavbvcvd!ve"vf'vg(vh
V
0
1
2
3
4
5
6
7
!8
"9
'10
(11
F
0
1
2
3
4
5
!6
"7
'8
(9
 
�
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
	regularization_losses
 
 
ge
VARIABLE_VALUEbatch_normalization_41/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_41/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_41/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_41/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

0
1
 
�
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
\Z
VARIABLE_VALUEdense_164/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_164/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
regularization_losses
\Z
VARIABLE_VALUEdense_165/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_165/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
\Z
VARIABLE_VALUEdense_166/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_166/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
 
�
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
#	variables
$trainable_variables
%regularization_losses
\Z
VARIABLE_VALUEdense_167/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_167/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 
�
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
)	variables
*trainable_variables
+regularization_losses
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

0
1
#
0
1
2
3
4

P0
 
 

0
1
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
	Qtotal
	Rcount
S	variables
T	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Q0
R1

S	variables
��
VARIABLE_VALUE#Adam/batch_normalization_41/gamma/mQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_41/beta/mPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_164/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_164/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_165/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_165/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_166/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_166/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_167/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_167/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_41/gamma/vQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_41/beta/vPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_164/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_164/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_165/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_165/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_166/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_166/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_167/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_167/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
,serving_default_batch_normalization_41_inputPlaceholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCall,serving_default_batch_normalization_41_input"batch_normalization_41/moving_mean&batch_normalization_41/moving_variancebatch_normalization_41/betabatch_normalization_41/gammadense_164/kerneldense_164/biasdense_165/kerneldense_165/biasdense_166/kerneldense_166/biasdense_167/kerneldense_167/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_6940463
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0batch_normalization_41/gamma/Read/ReadVariableOp/batch_normalization_41/beta/Read/ReadVariableOp6batch_normalization_41/moving_mean/Read/ReadVariableOp:batch_normalization_41/moving_variance/Read/ReadVariableOp$dense_164/kernel/Read/ReadVariableOp"dense_164/bias/Read/ReadVariableOp$dense_165/kernel/Read/ReadVariableOp"dense_165/bias/Read/ReadVariableOp$dense_166/kernel/Read/ReadVariableOp"dense_166/bias/Read/ReadVariableOp$dense_167/kernel/Read/ReadVariableOp"dense_167/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp7Adam/batch_normalization_41/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_41/beta/m/Read/ReadVariableOp+Adam/dense_164/kernel/m/Read/ReadVariableOp)Adam/dense_164/bias/m/Read/ReadVariableOp+Adam/dense_165/kernel/m/Read/ReadVariableOp)Adam/dense_165/bias/m/Read/ReadVariableOp+Adam/dense_166/kernel/m/Read/ReadVariableOp)Adam/dense_166/bias/m/Read/ReadVariableOp+Adam/dense_167/kernel/m/Read/ReadVariableOp)Adam/dense_167/bias/m/Read/ReadVariableOp7Adam/batch_normalization_41/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_41/beta/v/Read/ReadVariableOp+Adam/dense_164/kernel/v/Read/ReadVariableOp)Adam/dense_164/bias/v/Read/ReadVariableOp+Adam/dense_165/kernel/v/Read/ReadVariableOp)Adam/dense_165/bias/v/Read/ReadVariableOp+Adam/dense_166/kernel/v/Read/ReadVariableOp)Adam/dense_166/bias/v/Read/ReadVariableOp+Adam/dense_167/kernel/v/Read/ReadVariableOp)Adam/dense_167/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
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
GPU 2J 8� *)
f$R"
 __inference__traced_save_6940925
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization_41/gammabatch_normalization_41/beta"batch_normalization_41/moving_mean&batch_normalization_41/moving_variancedense_164/kerneldense_164/biasdense_165/kerneldense_165/biasdense_166/kerneldense_166/biasdense_167/kerneldense_167/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount#Adam/batch_normalization_41/gamma/m"Adam/batch_normalization_41/beta/mAdam/dense_164/kernel/mAdam/dense_164/bias/mAdam/dense_165/kernel/mAdam/dense_165/bias/mAdam/dense_166/kernel/mAdam/dense_166/bias/mAdam/dense_167/kernel/mAdam/dense_167/bias/m#Adam/batch_normalization_41/gamma/v"Adam/batch_normalization_41/beta/vAdam/dense_164/kernel/vAdam/dense_164/bias/vAdam/dense_165/kernel/vAdam/dense_165/bias/vAdam/dense_166/kernel/vAdam/dense_166/bias/vAdam/dense_167/kernel/vAdam/dense_167/bias/v*3
Tin,
*2(*
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
GPU 2J 8� *,
f'R%
#__inference__traced_restore_6941052��
�;
�

J__inference_sequential_41_layer_call_and_return_conditional_losses_6940567

inputsB
3batch_normalization_41_cast_readvariableop_resource:	�D
5batch_normalization_41_cast_1_readvariableop_resource:	�D
5batch_normalization_41_cast_2_readvariableop_resource:	�D
5batch_normalization_41_cast_3_readvariableop_resource:	�;
(dense_164_matmul_readvariableop_resource:	�7
)dense_164_biasadd_readvariableop_resource::
(dense_165_matmul_readvariableop_resource:7
)dense_165_biasadd_readvariableop_resource::
(dense_166_matmul_readvariableop_resource:7
)dense_166_biasadd_readvariableop_resource::
(dense_167_matmul_readvariableop_resource:7
)dense_167_biasadd_readvariableop_resource:
identity��*batch_normalization_41/Cast/ReadVariableOp�,batch_normalization_41/Cast_1/ReadVariableOp�,batch_normalization_41/Cast_2/ReadVariableOp�,batch_normalization_41/Cast_3/ReadVariableOp� dense_164/BiasAdd/ReadVariableOp�dense_164/MatMul/ReadVariableOp� dense_165/BiasAdd/ReadVariableOp�dense_165/MatMul/ReadVariableOp� dense_166/BiasAdd/ReadVariableOp�dense_166/MatMul/ReadVariableOp� dense_167/BiasAdd/ReadVariableOp�dense_167/MatMul/ReadVariableOp�
*batch_normalization_41/Cast/ReadVariableOpReadVariableOp3batch_normalization_41_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_41/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_41_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_41/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_41_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_41/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_41_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_41/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_41/batchnorm/addAddV24batch_normalization_41/Cast_1/ReadVariableOp:value:0/batch_normalization_41/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_41/batchnorm/RsqrtRsqrt(batch_normalization_41/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_41/batchnorm/mulMul*batch_normalization_41/batchnorm/Rsqrt:y:04batch_normalization_41/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_41/batchnorm/mul_1Mulinputs(batch_normalization_41/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
&batch_normalization_41/batchnorm/mul_2Mul2batch_normalization_41/Cast/ReadVariableOp:value:0(batch_normalization_41/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
$batch_normalization_41/batchnorm/subSub4batch_normalization_41/Cast_2/ReadVariableOp:value:0*batch_normalization_41/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_41/batchnorm/add_1AddV2*batch_normalization_41/batchnorm/mul_1:z:0(batch_normalization_41/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_164/MatMul/ReadVariableOpReadVariableOp(dense_164_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_164/MatMulMatMul*batch_normalization_41/batchnorm/add_1:z:0'dense_164/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_164/BiasAdd/ReadVariableOpReadVariableOp)dense_164_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_164/BiasAddBiasAdddense_164/MatMul:product:0(dense_164/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_165/MatMul/ReadVariableOpReadVariableOp(dense_165_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_165/MatMulMatMuldense_164/BiasAdd:output:0'dense_165/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_165/BiasAdd/ReadVariableOpReadVariableOp)dense_165_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_165/BiasAddBiasAdddense_165/MatMul:product:0(dense_165/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d
dense_165/ReluReludense_165/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_166/MatMul/ReadVariableOpReadVariableOp(dense_166_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_166/MatMulMatMuldense_165/Relu:activations:0'dense_166/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_166/BiasAdd/ReadVariableOpReadVariableOp)dense_166_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_166/BiasAddBiasAdddense_166/MatMul:product:0(dense_166/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d
dense_166/ReluReludense_166/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_167/MatMul/ReadVariableOpReadVariableOp(dense_167_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_167/MatMulMatMuldense_166/Relu:activations:0'dense_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_167/BiasAdd/ReadVariableOpReadVariableOp)dense_167_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_167/BiasAddBiasAdddense_167/MatMul:product:0(dense_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������i
IdentityIdentitydense_167/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp+^batch_normalization_41/Cast/ReadVariableOp-^batch_normalization_41/Cast_1/ReadVariableOp-^batch_normalization_41/Cast_2/ReadVariableOp-^batch_normalization_41/Cast_3/ReadVariableOp!^dense_164/BiasAdd/ReadVariableOp ^dense_164/MatMul/ReadVariableOp!^dense_165/BiasAdd/ReadVariableOp ^dense_165/MatMul/ReadVariableOp!^dense_166/BiasAdd/ReadVariableOp ^dense_166/MatMul/ReadVariableOp!^dense_167/BiasAdd/ReadVariableOp ^dense_167/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2X
*batch_normalization_41/Cast/ReadVariableOp*batch_normalization_41/Cast/ReadVariableOp2\
,batch_normalization_41/Cast_1/ReadVariableOp,batch_normalization_41/Cast_1/ReadVariableOp2\
,batch_normalization_41/Cast_2/ReadVariableOp,batch_normalization_41/Cast_2/ReadVariableOp2\
,batch_normalization_41/Cast_3/ReadVariableOp,batch_normalization_41/Cast_3/ReadVariableOp2D
 dense_164/BiasAdd/ReadVariableOp dense_164/BiasAdd/ReadVariableOp2B
dense_164/MatMul/ReadVariableOpdense_164/MatMul/ReadVariableOp2D
 dense_165/BiasAdd/ReadVariableOp dense_165/BiasAdd/ReadVariableOp2B
dense_165/MatMul/ReadVariableOpdense_165/MatMul/ReadVariableOp2D
 dense_166/BiasAdd/ReadVariableOp dense_166/BiasAdd/ReadVariableOp2B
dense_166/MatMul/ReadVariableOpdense_166/MatMul/ReadVariableOp2D
 dense_167/BiasAdd/ReadVariableOp dense_167/BiasAdd/ReadVariableOp2B
dense_167/MatMul/ReadVariableOpdense_167/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
J__inference_sequential_41_layer_call_and_return_conditional_losses_6940393 
batch_normalization_41_input-
batch_normalization_41_6940363:	�-
batch_normalization_41_6940365:	�-
batch_normalization_41_6940367:	�-
batch_normalization_41_6940369:	�$
dense_164_6940372:	�
dense_164_6940374:#
dense_165_6940377:
dense_165_6940379:#
dense_166_6940382:
dense_166_6940384:#
dense_167_6940387:
dense_167_6940389:
identity��.batch_normalization_41/StatefulPartitionedCall�!dense_164/StatefulPartitionedCall�!dense_165/StatefulPartitionedCall�!dense_166/StatefulPartitionedCall�!dense_167/StatefulPartitionedCall�
.batch_normalization_41/StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_41_inputbatch_normalization_41_6940363batch_normalization_41_6940365batch_normalization_41_6940367batch_normalization_41_6940369*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_6940032�
!dense_164/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_41/StatefulPartitionedCall:output:0dense_164_6940372dense_164_6940374*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_164_layer_call_and_return_conditional_losses_6940116�
!dense_165/StatefulPartitionedCallStatefulPartitionedCall*dense_164/StatefulPartitionedCall:output:0dense_165_6940377dense_165_6940379*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_165_layer_call_and_return_conditional_losses_6940133�
!dense_166/StatefulPartitionedCallStatefulPartitionedCall*dense_165/StatefulPartitionedCall:output:0dense_166_6940382dense_166_6940384*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_166_layer_call_and_return_conditional_losses_6940150�
!dense_167/StatefulPartitionedCallStatefulPartitionedCall*dense_166/StatefulPartitionedCall:output:0dense_167_6940387dense_167_6940389*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_167_layer_call_and_return_conditional_losses_6940166y
IdentityIdentity*dense_167/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_41/StatefulPartitionedCall"^dense_164/StatefulPartitionedCall"^dense_165/StatefulPartitionedCall"^dense_166/StatefulPartitionedCall"^dense_167/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2`
.batch_normalization_41/StatefulPartitionedCall.batch_normalization_41/StatefulPartitionedCall2F
!dense_164/StatefulPartitionedCall!dense_164/StatefulPartitionedCall2F
!dense_165/StatefulPartitionedCall!dense_165/StatefulPartitionedCall2F
!dense_166/StatefulPartitionedCall!dense_166/StatefulPartitionedCall2F
!dense_167/StatefulPartitionedCall!dense_167/StatefulPartitionedCall:f b
(
_output_shapes
:����������
6
_user_specified_namebatch_normalization_41_input
�
�
/__inference_sequential_41_layer_call_fn_6940200 
batch_normalization_41_input
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_41_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_41_layer_call_and_return_conditional_losses_6940173o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
(
_output_shapes
:����������
6
_user_specified_namebatch_normalization_41_input
�
�
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_6940673

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�-
cast_2_readvariableop_resource:	�-
cast_3_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpm
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������l
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�n
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�I
�
"__inference__wrapped_model_6940008 
batch_normalization_41_inputP
Asequential_41_batch_normalization_41_cast_readvariableop_resource:	�R
Csequential_41_batch_normalization_41_cast_1_readvariableop_resource:	�R
Csequential_41_batch_normalization_41_cast_2_readvariableop_resource:	�R
Csequential_41_batch_normalization_41_cast_3_readvariableop_resource:	�I
6sequential_41_dense_164_matmul_readvariableop_resource:	�E
7sequential_41_dense_164_biasadd_readvariableop_resource:H
6sequential_41_dense_165_matmul_readvariableop_resource:E
7sequential_41_dense_165_biasadd_readvariableop_resource:H
6sequential_41_dense_166_matmul_readvariableop_resource:E
7sequential_41_dense_166_biasadd_readvariableop_resource:H
6sequential_41_dense_167_matmul_readvariableop_resource:E
7sequential_41_dense_167_biasadd_readvariableop_resource:
identity��8sequential_41/batch_normalization_41/Cast/ReadVariableOp�:sequential_41/batch_normalization_41/Cast_1/ReadVariableOp�:sequential_41/batch_normalization_41/Cast_2/ReadVariableOp�:sequential_41/batch_normalization_41/Cast_3/ReadVariableOp�.sequential_41/dense_164/BiasAdd/ReadVariableOp�-sequential_41/dense_164/MatMul/ReadVariableOp�.sequential_41/dense_165/BiasAdd/ReadVariableOp�-sequential_41/dense_165/MatMul/ReadVariableOp�.sequential_41/dense_166/BiasAdd/ReadVariableOp�-sequential_41/dense_166/MatMul/ReadVariableOp�.sequential_41/dense_167/BiasAdd/ReadVariableOp�-sequential_41/dense_167/MatMul/ReadVariableOp�
8sequential_41/batch_normalization_41/Cast/ReadVariableOpReadVariableOpAsequential_41_batch_normalization_41_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:sequential_41/batch_normalization_41/Cast_1/ReadVariableOpReadVariableOpCsequential_41_batch_normalization_41_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:sequential_41/batch_normalization_41/Cast_2/ReadVariableOpReadVariableOpCsequential_41_batch_normalization_41_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:sequential_41/batch_normalization_41/Cast_3/ReadVariableOpReadVariableOpCsequential_41_batch_normalization_41_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0y
4sequential_41/batch_normalization_41/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
2sequential_41/batch_normalization_41/batchnorm/addAddV2Bsequential_41/batch_normalization_41/Cast_1/ReadVariableOp:value:0=sequential_41/batch_normalization_41/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
4sequential_41/batch_normalization_41/batchnorm/RsqrtRsqrt6sequential_41/batch_normalization_41/batchnorm/add:z:0*
T0*
_output_shapes	
:��
2sequential_41/batch_normalization_41/batchnorm/mulMul8sequential_41/batch_normalization_41/batchnorm/Rsqrt:y:0Bsequential_41/batch_normalization_41/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
4sequential_41/batch_normalization_41/batchnorm/mul_1Mulbatch_normalization_41_input6sequential_41/batch_normalization_41/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
4sequential_41/batch_normalization_41/batchnorm/mul_2Mul@sequential_41/batch_normalization_41/Cast/ReadVariableOp:value:06sequential_41/batch_normalization_41/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
2sequential_41/batch_normalization_41/batchnorm/subSubBsequential_41/batch_normalization_41/Cast_2/ReadVariableOp:value:08sequential_41/batch_normalization_41/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
4sequential_41/batch_normalization_41/batchnorm/add_1AddV28sequential_41/batch_normalization_41/batchnorm/mul_1:z:06sequential_41/batch_normalization_41/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
-sequential_41/dense_164/MatMul/ReadVariableOpReadVariableOp6sequential_41_dense_164_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_41/dense_164/MatMulMatMul8sequential_41/batch_normalization_41/batchnorm/add_1:z:05sequential_41/dense_164/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
.sequential_41/dense_164/BiasAdd/ReadVariableOpReadVariableOp7sequential_41_dense_164_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_41/dense_164/BiasAddBiasAdd(sequential_41/dense_164/MatMul:product:06sequential_41/dense_164/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-sequential_41/dense_165/MatMul/ReadVariableOpReadVariableOp6sequential_41_dense_165_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
sequential_41/dense_165/MatMulMatMul(sequential_41/dense_164/BiasAdd:output:05sequential_41/dense_165/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
.sequential_41/dense_165/BiasAdd/ReadVariableOpReadVariableOp7sequential_41_dense_165_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_41/dense_165/BiasAddBiasAdd(sequential_41/dense_165/MatMul:product:06sequential_41/dense_165/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
sequential_41/dense_165/ReluRelu(sequential_41/dense_165/BiasAdd:output:0*
T0*'
_output_shapes
:����������
-sequential_41/dense_166/MatMul/ReadVariableOpReadVariableOp6sequential_41_dense_166_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
sequential_41/dense_166/MatMulMatMul*sequential_41/dense_165/Relu:activations:05sequential_41/dense_166/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
.sequential_41/dense_166/BiasAdd/ReadVariableOpReadVariableOp7sequential_41_dense_166_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_41/dense_166/BiasAddBiasAdd(sequential_41/dense_166/MatMul:product:06sequential_41/dense_166/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
sequential_41/dense_166/ReluRelu(sequential_41/dense_166/BiasAdd:output:0*
T0*'
_output_shapes
:����������
-sequential_41/dense_167/MatMul/ReadVariableOpReadVariableOp6sequential_41_dense_167_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
sequential_41/dense_167/MatMulMatMul*sequential_41/dense_166/Relu:activations:05sequential_41/dense_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
.sequential_41/dense_167/BiasAdd/ReadVariableOpReadVariableOp7sequential_41_dense_167_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_41/dense_167/BiasAddBiasAdd(sequential_41/dense_167/MatMul:product:06sequential_41/dense_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w
IdentityIdentity(sequential_41/dense_167/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp9^sequential_41/batch_normalization_41/Cast/ReadVariableOp;^sequential_41/batch_normalization_41/Cast_1/ReadVariableOp;^sequential_41/batch_normalization_41/Cast_2/ReadVariableOp;^sequential_41/batch_normalization_41/Cast_3/ReadVariableOp/^sequential_41/dense_164/BiasAdd/ReadVariableOp.^sequential_41/dense_164/MatMul/ReadVariableOp/^sequential_41/dense_165/BiasAdd/ReadVariableOp.^sequential_41/dense_165/MatMul/ReadVariableOp/^sequential_41/dense_166/BiasAdd/ReadVariableOp.^sequential_41/dense_166/MatMul/ReadVariableOp/^sequential_41/dense_167/BiasAdd/ReadVariableOp.^sequential_41/dense_167/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2t
8sequential_41/batch_normalization_41/Cast/ReadVariableOp8sequential_41/batch_normalization_41/Cast/ReadVariableOp2x
:sequential_41/batch_normalization_41/Cast_1/ReadVariableOp:sequential_41/batch_normalization_41/Cast_1/ReadVariableOp2x
:sequential_41/batch_normalization_41/Cast_2/ReadVariableOp:sequential_41/batch_normalization_41/Cast_2/ReadVariableOp2x
:sequential_41/batch_normalization_41/Cast_3/ReadVariableOp:sequential_41/batch_normalization_41/Cast_3/ReadVariableOp2`
.sequential_41/dense_164/BiasAdd/ReadVariableOp.sequential_41/dense_164/BiasAdd/ReadVariableOp2^
-sequential_41/dense_164/MatMul/ReadVariableOp-sequential_41/dense_164/MatMul/ReadVariableOp2`
.sequential_41/dense_165/BiasAdd/ReadVariableOp.sequential_41/dense_165/BiasAdd/ReadVariableOp2^
-sequential_41/dense_165/MatMul/ReadVariableOp-sequential_41/dense_165/MatMul/ReadVariableOp2`
.sequential_41/dense_166/BiasAdd/ReadVariableOp.sequential_41/dense_166/BiasAdd/ReadVariableOp2^
-sequential_41/dense_166/MatMul/ReadVariableOp-sequential_41/dense_166/MatMul/ReadVariableOp2`
.sequential_41/dense_167/BiasAdd/ReadVariableOp.sequential_41/dense_167/BiasAdd/ReadVariableOp2^
-sequential_41/dense_167/MatMul/ReadVariableOp-sequential_41/dense_167/MatMul/ReadVariableOp:f b
(
_output_shapes
:����������
6
_user_specified_namebatch_normalization_41_input
�

�
/__inference_sequential_41_layer_call_fn_6940492

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_41_layer_call_and_return_conditional_losses_6940173o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
F__inference_dense_166_layer_call_and_return_conditional_losses_6940766

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�S
�
 __inference__traced_save_6940925
file_prefix;
7savev2_batch_normalization_41_gamma_read_readvariableop:
6savev2_batch_normalization_41_beta_read_readvariableopA
=savev2_batch_normalization_41_moving_mean_read_readvariableopE
Asavev2_batch_normalization_41_moving_variance_read_readvariableop/
+savev2_dense_164_kernel_read_readvariableop-
)savev2_dense_164_bias_read_readvariableop/
+savev2_dense_165_kernel_read_readvariableop-
)savev2_dense_165_bias_read_readvariableop/
+savev2_dense_166_kernel_read_readvariableop-
)savev2_dense_166_bias_read_readvariableop/
+savev2_dense_167_kernel_read_readvariableop-
)savev2_dense_167_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopB
>savev2_adam_batch_normalization_41_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_41_beta_m_read_readvariableop6
2savev2_adam_dense_164_kernel_m_read_readvariableop4
0savev2_adam_dense_164_bias_m_read_readvariableop6
2savev2_adam_dense_165_kernel_m_read_readvariableop4
0savev2_adam_dense_165_bias_m_read_readvariableop6
2savev2_adam_dense_166_kernel_m_read_readvariableop4
0savev2_adam_dense_166_bias_m_read_readvariableop6
2savev2_adam_dense_167_kernel_m_read_readvariableop4
0savev2_adam_dense_167_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_41_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_41_beta_v_read_readvariableop6
2savev2_adam_dense_164_kernel_v_read_readvariableop4
0savev2_adam_dense_164_bias_v_read_readvariableop6
2savev2_adam_dense_165_kernel_v_read_readvariableop4
0savev2_adam_dense_165_bias_v_read_readvariableop6
2savev2_adam_dense_166_kernel_v_read_readvariableop4
0savev2_adam_dense_166_bias_v_read_readvariableop6
2savev2_adam_dense_167_kernel_v_read_readvariableop4
0savev2_adam_dense_167_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*�
value�B�(B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:07savev2_batch_normalization_41_gamma_read_readvariableop6savev2_batch_normalization_41_beta_read_readvariableop=savev2_batch_normalization_41_moving_mean_read_readvariableopAsavev2_batch_normalization_41_moving_variance_read_readvariableop+savev2_dense_164_kernel_read_readvariableop)savev2_dense_164_bias_read_readvariableop+savev2_dense_165_kernel_read_readvariableop)savev2_dense_165_bias_read_readvariableop+savev2_dense_166_kernel_read_readvariableop)savev2_dense_166_bias_read_readvariableop+savev2_dense_167_kernel_read_readvariableop)savev2_dense_167_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop>savev2_adam_batch_normalization_41_gamma_m_read_readvariableop=savev2_adam_batch_normalization_41_beta_m_read_readvariableop2savev2_adam_dense_164_kernel_m_read_readvariableop0savev2_adam_dense_164_bias_m_read_readvariableop2savev2_adam_dense_165_kernel_m_read_readvariableop0savev2_adam_dense_165_bias_m_read_readvariableop2savev2_adam_dense_166_kernel_m_read_readvariableop0savev2_adam_dense_166_bias_m_read_readvariableop2savev2_adam_dense_167_kernel_m_read_readvariableop0savev2_adam_dense_167_bias_m_read_readvariableop>savev2_adam_batch_normalization_41_gamma_v_read_readvariableop=savev2_adam_batch_normalization_41_beta_v_read_readvariableop2savev2_adam_dense_164_kernel_v_read_readvariableop0savev2_adam_dense_164_bias_v_read_readvariableop2savev2_adam_dense_165_kernel_v_read_readvariableop0savev2_adam_dense_165_bias_v_read_readvariableop2savev2_adam_dense_166_kernel_v_read_readvariableop0savev2_adam_dense_166_bias_v_read_readvariableop2savev2_adam_dense_167_kernel_v_read_readvariableop0savev2_adam_dense_167_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :�:�:�:�:	�:::::::: : : : : : : :�:�:	�::::::::�:�:	�:::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:�:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::!

_output_shapes	
:�:!

_output_shapes	
:�:% !

_output_shapes
:	�: !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
::$$ 

_output_shapes

:: %

_output_shapes
::$& 

_output_shapes

:: '

_output_shapes
::(

_output_shapes
: 
�
�
8__inference_batch_normalization_41_layer_call_fn_6940653

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_6940079p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
F__inference_dense_164_layer_call_and_return_conditional_losses_6940116

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_dense_167_layer_call_fn_6940775

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_167_layer_call_and_return_conditional_losses_6940166o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_dense_165_layer_call_fn_6940735

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_165_layer_call_and_return_conditional_losses_6940133o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
F__inference_dense_165_layer_call_and_return_conditional_losses_6940746

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
#__inference__traced_restore_6941052
file_prefix<
-assignvariableop_batch_normalization_41_gamma:	�=
.assignvariableop_1_batch_normalization_41_beta:	�D
5assignvariableop_2_batch_normalization_41_moving_mean:	�H
9assignvariableop_3_batch_normalization_41_moving_variance:	�6
#assignvariableop_4_dense_164_kernel:	�/
!assignvariableop_5_dense_164_bias:5
#assignvariableop_6_dense_165_kernel:/
!assignvariableop_7_dense_165_bias:5
#assignvariableop_8_dense_166_kernel:/
!assignvariableop_9_dense_166_bias:6
$assignvariableop_10_dense_167_kernel:0
"assignvariableop_11_dense_167_bias:'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: #
assignvariableop_17_total: #
assignvariableop_18_count: F
7assignvariableop_19_adam_batch_normalization_41_gamma_m:	�E
6assignvariableop_20_adam_batch_normalization_41_beta_m:	�>
+assignvariableop_21_adam_dense_164_kernel_m:	�7
)assignvariableop_22_adam_dense_164_bias_m:=
+assignvariableop_23_adam_dense_165_kernel_m:7
)assignvariableop_24_adam_dense_165_bias_m:=
+assignvariableop_25_adam_dense_166_kernel_m:7
)assignvariableop_26_adam_dense_166_bias_m:=
+assignvariableop_27_adam_dense_167_kernel_m:7
)assignvariableop_28_adam_dense_167_bias_m:F
7assignvariableop_29_adam_batch_normalization_41_gamma_v:	�E
6assignvariableop_30_adam_batch_normalization_41_beta_v:	�>
+assignvariableop_31_adam_dense_164_kernel_v:	�7
)assignvariableop_32_adam_dense_164_bias_v:=
+assignvariableop_33_adam_dense_165_kernel_v:7
)assignvariableop_34_adam_dense_165_bias_v:=
+assignvariableop_35_adam_dense_166_kernel_v:7
)assignvariableop_36_adam_dense_166_bias_v:=
+assignvariableop_37_adam_dense_167_kernel_v:7
)assignvariableop_38_adam_dense_167_bias_v:
identity_40��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*�
value�B�(B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp-assignvariableop_batch_normalization_41_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp.assignvariableop_1_batch_normalization_41_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp5assignvariableop_2_batch_normalization_41_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp9assignvariableop_3_batch_normalization_41_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_164_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_164_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_165_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_165_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp#assignvariableop_8_dense_166_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_166_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp$assignvariableop_10_dense_167_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp"assignvariableop_11_dense_167_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp7assignvariableop_19_adam_batch_normalization_41_gamma_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_batch_normalization_41_beta_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dense_164_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_164_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_165_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_165_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_dense_166_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_166_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_167_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_167_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp7assignvariableop_29_adam_batch_normalization_41_gamma_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp6assignvariableop_30_adam_batch_normalization_41_beta_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_164_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_164_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_dense_165_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_dense_165_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_dense_166_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_dense_166_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_dense_167_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_dense_167_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_40IdentityIdentity_39:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_40Identity_40:output:0*c
_input_shapesR
P: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
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
�

�
F__inference_dense_165_layer_call_and_return_conditional_losses_6940133

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
F__inference_dense_166_layer_call_and_return_conditional_losses_6940150

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_sequential_41_layer_call_and_return_conditional_losses_6940426 
batch_normalization_41_input-
batch_normalization_41_6940396:	�-
batch_normalization_41_6940398:	�-
batch_normalization_41_6940400:	�-
batch_normalization_41_6940402:	�$
dense_164_6940405:	�
dense_164_6940407:#
dense_165_6940410:
dense_165_6940412:#
dense_166_6940415:
dense_166_6940417:#
dense_167_6940420:
dense_167_6940422:
identity��.batch_normalization_41/StatefulPartitionedCall�!dense_164/StatefulPartitionedCall�!dense_165/StatefulPartitionedCall�!dense_166/StatefulPartitionedCall�!dense_167/StatefulPartitionedCall�
.batch_normalization_41/StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_41_inputbatch_normalization_41_6940396batch_normalization_41_6940398batch_normalization_41_6940400batch_normalization_41_6940402*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_6940079�
!dense_164/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_41/StatefulPartitionedCall:output:0dense_164_6940405dense_164_6940407*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_164_layer_call_and_return_conditional_losses_6940116�
!dense_165/StatefulPartitionedCallStatefulPartitionedCall*dense_164/StatefulPartitionedCall:output:0dense_165_6940410dense_165_6940412*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_165_layer_call_and_return_conditional_losses_6940133�
!dense_166/StatefulPartitionedCallStatefulPartitionedCall*dense_165/StatefulPartitionedCall:output:0dense_166_6940415dense_166_6940417*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_166_layer_call_and_return_conditional_losses_6940150�
!dense_167/StatefulPartitionedCallStatefulPartitionedCall*dense_166/StatefulPartitionedCall:output:0dense_167_6940420dense_167_6940422*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_167_layer_call_and_return_conditional_losses_6940166y
IdentityIdentity*dense_167/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_41/StatefulPartitionedCall"^dense_164/StatefulPartitionedCall"^dense_165/StatefulPartitionedCall"^dense_166/StatefulPartitionedCall"^dense_167/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2`
.batch_normalization_41/StatefulPartitionedCall.batch_normalization_41/StatefulPartitionedCall2F
!dense_164/StatefulPartitionedCall!dense_164/StatefulPartitionedCall2F
!dense_165/StatefulPartitionedCall!dense_165/StatefulPartitionedCall2F
!dense_166/StatefulPartitionedCall!dense_166/StatefulPartitionedCall2F
!dense_167/StatefulPartitionedCall!dense_167/StatefulPartitionedCall:f b
(
_output_shapes
:����������
6
_user_specified_namebatch_normalization_41_input
�
�
8__inference_batch_normalization_41_layer_call_fn_6940640

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_6940032p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
/__inference_sequential_41_layer_call_fn_6940360 
batch_normalization_41_input
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_41_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_41_layer_call_and_return_conditional_losses_6940304o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
(
_output_shapes
:����������
6
_user_specified_namebatch_normalization_41_input
�U
�
J__inference_sequential_41_layer_call_and_return_conditional_losses_6940627

inputsM
>batch_normalization_41_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_41_assignmovingavg_1_readvariableop_resource:	�B
3batch_normalization_41_cast_readvariableop_resource:	�D
5batch_normalization_41_cast_1_readvariableop_resource:	�;
(dense_164_matmul_readvariableop_resource:	�7
)dense_164_biasadd_readvariableop_resource::
(dense_165_matmul_readvariableop_resource:7
)dense_165_biasadd_readvariableop_resource::
(dense_166_matmul_readvariableop_resource:7
)dense_166_biasadd_readvariableop_resource::
(dense_167_matmul_readvariableop_resource:7
)dense_167_biasadd_readvariableop_resource:
identity��&batch_normalization_41/AssignMovingAvg�5batch_normalization_41/AssignMovingAvg/ReadVariableOp�(batch_normalization_41/AssignMovingAvg_1�7batch_normalization_41/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_41/Cast/ReadVariableOp�,batch_normalization_41/Cast_1/ReadVariableOp� dense_164/BiasAdd/ReadVariableOp�dense_164/MatMul/ReadVariableOp� dense_165/BiasAdd/ReadVariableOp�dense_165/MatMul/ReadVariableOp� dense_166/BiasAdd/ReadVariableOp�dense_166/MatMul/ReadVariableOp� dense_167/BiasAdd/ReadVariableOp�dense_167/MatMul/ReadVariableOp
5batch_normalization_41/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_41/moments/meanMeaninputs>batch_normalization_41/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_41/moments/StopGradientStopGradient,batch_normalization_41/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_41/moments/SquaredDifferenceSquaredDifferenceinputs4batch_normalization_41/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
9batch_normalization_41/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_41/moments/varianceMean4batch_normalization_41/moments/SquaredDifference:z:0Bbatch_normalization_41/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_41/moments/SqueezeSqueeze,batch_normalization_41/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_41/moments/Squeeze_1Squeeze0batch_normalization_41/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_41/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_41/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_41_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_41/AssignMovingAvg/subSub=batch_normalization_41/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_41/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_41/AssignMovingAvg/mulMul.batch_normalization_41/AssignMovingAvg/sub:z:05batch_normalization_41/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_41/AssignMovingAvgAssignSubVariableOp>batch_normalization_41_assignmovingavg_readvariableop_resource.batch_normalization_41/AssignMovingAvg/mul:z:06^batch_normalization_41/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_41/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_41/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_41_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_41/AssignMovingAvg_1/subSub?batch_normalization_41/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_41/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_41/AssignMovingAvg_1/mulMul0batch_normalization_41/AssignMovingAvg_1/sub:z:07batch_normalization_41/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_41/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_41_assignmovingavg_1_readvariableop_resource0batch_normalization_41/AssignMovingAvg_1/mul:z:08^batch_normalization_41/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
*batch_normalization_41/Cast/ReadVariableOpReadVariableOp3batch_normalization_41_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_41/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_41_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_41/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_41/batchnorm/addAddV21batch_normalization_41/moments/Squeeze_1:output:0/batch_normalization_41/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_41/batchnorm/RsqrtRsqrt(batch_normalization_41/batchnorm/add:z:0*
T0*
_output_shapes	
:��
$batch_normalization_41/batchnorm/mulMul*batch_normalization_41/batchnorm/Rsqrt:y:04batch_normalization_41/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_41/batchnorm/mul_1Mulinputs(batch_normalization_41/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
&batch_normalization_41/batchnorm/mul_2Mul/batch_normalization_41/moments/Squeeze:output:0(batch_normalization_41/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
$batch_normalization_41/batchnorm/subSub2batch_normalization_41/Cast/ReadVariableOp:value:0*batch_normalization_41/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_41/batchnorm/add_1AddV2*batch_normalization_41/batchnorm/mul_1:z:0(batch_normalization_41/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_164/MatMul/ReadVariableOpReadVariableOp(dense_164_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_164/MatMulMatMul*batch_normalization_41/batchnorm/add_1:z:0'dense_164/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_164/BiasAdd/ReadVariableOpReadVariableOp)dense_164_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_164/BiasAddBiasAdddense_164/MatMul:product:0(dense_164/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_165/MatMul/ReadVariableOpReadVariableOp(dense_165_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_165/MatMulMatMuldense_164/BiasAdd:output:0'dense_165/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_165/BiasAdd/ReadVariableOpReadVariableOp)dense_165_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_165/BiasAddBiasAdddense_165/MatMul:product:0(dense_165/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d
dense_165/ReluReludense_165/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_166/MatMul/ReadVariableOpReadVariableOp(dense_166_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_166/MatMulMatMuldense_165/Relu:activations:0'dense_166/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_166/BiasAdd/ReadVariableOpReadVariableOp)dense_166_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_166/BiasAddBiasAdddense_166/MatMul:product:0(dense_166/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d
dense_166/ReluReludense_166/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_167/MatMul/ReadVariableOpReadVariableOp(dense_167_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_167/MatMulMatMuldense_166/Relu:activations:0'dense_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_167/BiasAdd/ReadVariableOpReadVariableOp)dense_167_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_167/BiasAddBiasAdddense_167/MatMul:product:0(dense_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������i
IdentityIdentitydense_167/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^batch_normalization_41/AssignMovingAvg6^batch_normalization_41/AssignMovingAvg/ReadVariableOp)^batch_normalization_41/AssignMovingAvg_18^batch_normalization_41/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_41/Cast/ReadVariableOp-^batch_normalization_41/Cast_1/ReadVariableOp!^dense_164/BiasAdd/ReadVariableOp ^dense_164/MatMul/ReadVariableOp!^dense_165/BiasAdd/ReadVariableOp ^dense_165/MatMul/ReadVariableOp!^dense_166/BiasAdd/ReadVariableOp ^dense_166/MatMul/ReadVariableOp!^dense_167/BiasAdd/ReadVariableOp ^dense_167/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2P
&batch_normalization_41/AssignMovingAvg&batch_normalization_41/AssignMovingAvg2n
5batch_normalization_41/AssignMovingAvg/ReadVariableOp5batch_normalization_41/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_41/AssignMovingAvg_1(batch_normalization_41/AssignMovingAvg_12r
7batch_normalization_41/AssignMovingAvg_1/ReadVariableOp7batch_normalization_41/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_41/Cast/ReadVariableOp*batch_normalization_41/Cast/ReadVariableOp2\
,batch_normalization_41/Cast_1/ReadVariableOp,batch_normalization_41/Cast_1/ReadVariableOp2D
 dense_164/BiasAdd/ReadVariableOp dense_164/BiasAdd/ReadVariableOp2B
dense_164/MatMul/ReadVariableOpdense_164/MatMul/ReadVariableOp2D
 dense_165/BiasAdd/ReadVariableOp dense_165/BiasAdd/ReadVariableOp2B
dense_165/MatMul/ReadVariableOpdense_165/MatMul/ReadVariableOp2D
 dense_166/BiasAdd/ReadVariableOp dense_166/BiasAdd/ReadVariableOp2B
dense_166/MatMul/ReadVariableOpdense_166/MatMul/ReadVariableOp2D
 dense_167/BiasAdd/ReadVariableOp dense_167/BiasAdd/ReadVariableOp2B
dense_167/MatMul/ReadVariableOpdense_167/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_dense_166_layer_call_fn_6940755

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_166_layer_call_and_return_conditional_losses_6940150o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
F__inference_dense_167_layer_call_and_return_conditional_losses_6940785

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_dense_164_layer_call_fn_6940716

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_164_layer_call_and_return_conditional_losses_6940116o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�$
�
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_6940079

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0m
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�l
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
F__inference_dense_164_layer_call_and_return_conditional_losses_6940726

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�$
�
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_6940707

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0m
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�l
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
F__inference_dense_167_layer_call_and_return_conditional_losses_6940166

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_6940032

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�-
cast_2_readvariableop_resource:	�-
cast_3_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpm
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������l
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�n
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_6940463 
batch_normalization_41_input
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_41_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_6940008o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
(
_output_shapes
:����������
6
_user_specified_namebatch_normalization_41_input
�

�
/__inference_sequential_41_layer_call_fn_6940521

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_41_layer_call_and_return_conditional_losses_6940304o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
J__inference_sequential_41_layer_call_and_return_conditional_losses_6940304

inputs-
batch_normalization_41_6940274:	�-
batch_normalization_41_6940276:	�-
batch_normalization_41_6940278:	�-
batch_normalization_41_6940280:	�$
dense_164_6940283:	�
dense_164_6940285:#
dense_165_6940288:
dense_165_6940290:#
dense_166_6940293:
dense_166_6940295:#
dense_167_6940298:
dense_167_6940300:
identity��.batch_normalization_41/StatefulPartitionedCall�!dense_164/StatefulPartitionedCall�!dense_165/StatefulPartitionedCall�!dense_166/StatefulPartitionedCall�!dense_167/StatefulPartitionedCall�
.batch_normalization_41/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_41_6940274batch_normalization_41_6940276batch_normalization_41_6940278batch_normalization_41_6940280*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_6940079�
!dense_164/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_41/StatefulPartitionedCall:output:0dense_164_6940283dense_164_6940285*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_164_layer_call_and_return_conditional_losses_6940116�
!dense_165/StatefulPartitionedCallStatefulPartitionedCall*dense_164/StatefulPartitionedCall:output:0dense_165_6940288dense_165_6940290*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_165_layer_call_and_return_conditional_losses_6940133�
!dense_166/StatefulPartitionedCallStatefulPartitionedCall*dense_165/StatefulPartitionedCall:output:0dense_166_6940293dense_166_6940295*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_166_layer_call_and_return_conditional_losses_6940150�
!dense_167/StatefulPartitionedCallStatefulPartitionedCall*dense_166/StatefulPartitionedCall:output:0dense_167_6940298dense_167_6940300*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_167_layer_call_and_return_conditional_losses_6940166y
IdentityIdentity*dense_167/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_41/StatefulPartitionedCall"^dense_164/StatefulPartitionedCall"^dense_165/StatefulPartitionedCall"^dense_166/StatefulPartitionedCall"^dense_167/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2`
.batch_normalization_41/StatefulPartitionedCall.batch_normalization_41/StatefulPartitionedCall2F
!dense_164/StatefulPartitionedCall!dense_164/StatefulPartitionedCall2F
!dense_165/StatefulPartitionedCall!dense_165/StatefulPartitionedCall2F
!dense_166/StatefulPartitionedCall!dense_166/StatefulPartitionedCall2F
!dense_167/StatefulPartitionedCall!dense_167/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
J__inference_sequential_41_layer_call_and_return_conditional_losses_6940173

inputs-
batch_normalization_41_6940097:	�-
batch_normalization_41_6940099:	�-
batch_normalization_41_6940101:	�-
batch_normalization_41_6940103:	�$
dense_164_6940117:	�
dense_164_6940119:#
dense_165_6940134:
dense_165_6940136:#
dense_166_6940151:
dense_166_6940153:#
dense_167_6940167:
dense_167_6940169:
identity��.batch_normalization_41/StatefulPartitionedCall�!dense_164/StatefulPartitionedCall�!dense_165/StatefulPartitionedCall�!dense_166/StatefulPartitionedCall�!dense_167/StatefulPartitionedCall�
.batch_normalization_41/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_41_6940097batch_normalization_41_6940099batch_normalization_41_6940101batch_normalization_41_6940103*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_6940032�
!dense_164/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_41/StatefulPartitionedCall:output:0dense_164_6940117dense_164_6940119*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_164_layer_call_and_return_conditional_losses_6940116�
!dense_165/StatefulPartitionedCallStatefulPartitionedCall*dense_164/StatefulPartitionedCall:output:0dense_165_6940134dense_165_6940136*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_165_layer_call_and_return_conditional_losses_6940133�
!dense_166/StatefulPartitionedCallStatefulPartitionedCall*dense_165/StatefulPartitionedCall:output:0dense_166_6940151dense_166_6940153*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_166_layer_call_and_return_conditional_losses_6940150�
!dense_167/StatefulPartitionedCallStatefulPartitionedCall*dense_166/StatefulPartitionedCall:output:0dense_167_6940167dense_167_6940169*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_167_layer_call_and_return_conditional_losses_6940166y
IdentityIdentity*dense_167/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_41/StatefulPartitionedCall"^dense_164/StatefulPartitionedCall"^dense_165/StatefulPartitionedCall"^dense_166/StatefulPartitionedCall"^dense_167/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2`
.batch_normalization_41/StatefulPartitionedCall.batch_normalization_41/StatefulPartitionedCall2F
!dense_164/StatefulPartitionedCall!dense_164/StatefulPartitionedCall2F
!dense_165/StatefulPartitionedCall!dense_165/StatefulPartitionedCall2F
!dense_166/StatefulPartitionedCall!dense_166/StatefulPartitionedCall2F
!dense_167/StatefulPartitionedCall!dense_167/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
f
batch_normalization_41_inputF
.serving_default_batch_normalization_41_input:0����������=
	dense_1670
StatefulPartitionedCall:0���������tensorflow/serving/predict:�s
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
i__call__
*j&call_and_return_all_conditional_losses
k_default_save_signature"
_tf_keras_sequential
�
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
l__call__
*m&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
p__call__
*q&call_and_return_all_conditional_losses"
_tf_keras_layer
�

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
r__call__
*s&call_and_return_all_conditional_losses"
_tf_keras_layer
�

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
�
-iter

.beta_1

/beta_2
	0decay
1learning_ratemUmVmWmXmYmZ!m["m\'m](m^v_v`vavbvcvd!ve"vf'vg(vh"
	optimizer
v
0
1
2
3
4
5
6
7
!8
"9
'10
(11"
trackable_list_wrapper
f
0
1
2
3
4
5
!6
"7
'8
(9"
trackable_list_wrapper
 "
trackable_list_wrapper
�
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
	regularization_losses
i__call__
k_default_save_signature
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
,
vserving_default"
signature_map
 "
trackable_list_wrapper
+:)�2batch_normalization_41/gamma
*:(�2batch_normalization_41/beta
3:1� (2"batch_normalization_41/moving_mean
7:5� (2&batch_normalization_41/moving_variance
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
#:!	�2dense_164/kernel
:2dense_164/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
regularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
": 2dense_165/kernel
:2dense_165/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
": 2dense_166/kernel
:2dense_166/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
#	variables
$trainable_variables
%regularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
": 2dense_167/kernel
:2dense_167/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
)	variables
*trainable_variables
+regularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.
0
1"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
'
P0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
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
 "
trackable_dict_wrapper
N
	Qtotal
	Rcount
S	variables
T	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
Q0
R1"
trackable_list_wrapper
-
S	variables"
_generic_user_object
0:.�2#Adam/batch_normalization_41/gamma/m
/:-�2"Adam/batch_normalization_41/beta/m
(:&	�2Adam/dense_164/kernel/m
!:2Adam/dense_164/bias/m
':%2Adam/dense_165/kernel/m
!:2Adam/dense_165/bias/m
':%2Adam/dense_166/kernel/m
!:2Adam/dense_166/bias/m
':%2Adam/dense_167/kernel/m
!:2Adam/dense_167/bias/m
0:.�2#Adam/batch_normalization_41/gamma/v
/:-�2"Adam/batch_normalization_41/beta/v
(:&	�2Adam/dense_164/kernel/v
!:2Adam/dense_164/bias/v
':%2Adam/dense_165/kernel/v
!:2Adam/dense_165/bias/v
':%2Adam/dense_166/kernel/v
!:2Adam/dense_166/bias/v
':%2Adam/dense_167/kernel/v
!:2Adam/dense_167/bias/v
�2�
/__inference_sequential_41_layer_call_fn_6940200
/__inference_sequential_41_layer_call_fn_6940492
/__inference_sequential_41_layer_call_fn_6940521
/__inference_sequential_41_layer_call_fn_6940360�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
J__inference_sequential_41_layer_call_and_return_conditional_losses_6940567
J__inference_sequential_41_layer_call_and_return_conditional_losses_6940627
J__inference_sequential_41_layer_call_and_return_conditional_losses_6940393
J__inference_sequential_41_layer_call_and_return_conditional_losses_6940426�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
"__inference__wrapped_model_6940008batch_normalization_41_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
8__inference_batch_normalization_41_layer_call_fn_6940640
8__inference_batch_normalization_41_layer_call_fn_6940653�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_6940673
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_6940707�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
+__inference_dense_164_layer_call_fn_6940716�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_dense_164_layer_call_and_return_conditional_losses_6940726�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_dense_165_layer_call_fn_6940735�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_dense_165_layer_call_and_return_conditional_losses_6940746�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_dense_166_layer_call_fn_6940755�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_dense_166_layer_call_and_return_conditional_losses_6940766�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_dense_167_layer_call_fn_6940775�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_dense_167_layer_call_and_return_conditional_losses_6940785�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_6940463batch_normalization_41_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
"__inference__wrapped_model_6940008�!"'(F�C
<�9
7�4
batch_normalization_41_input����������
� "5�2
0
	dense_167#� 
	dense_167����������
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_6940673d4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_6940707d4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
8__inference_batch_normalization_41_layer_call_fn_6940640W4�1
*�'
!�
inputs����������
p 
� "������������
8__inference_batch_normalization_41_layer_call_fn_6940653W4�1
*�'
!�
inputs����������
p
� "������������
F__inference_dense_164_layer_call_and_return_conditional_losses_6940726]0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� 
+__inference_dense_164_layer_call_fn_6940716P0�-
&�#
!�
inputs����������
� "�����������
F__inference_dense_165_layer_call_and_return_conditional_losses_6940746\/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� ~
+__inference_dense_165_layer_call_fn_6940735O/�,
%�"
 �
inputs���������
� "�����������
F__inference_dense_166_layer_call_and_return_conditional_losses_6940766\!"/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� ~
+__inference_dense_166_layer_call_fn_6940755O!"/�,
%�"
 �
inputs���������
� "�����������
F__inference_dense_167_layer_call_and_return_conditional_losses_6940785\'(/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� ~
+__inference_dense_167_layer_call_fn_6940775O'(/�,
%�"
 �
inputs���������
� "�����������
J__inference_sequential_41_layer_call_and_return_conditional_losses_6940393�!"'(N�K
D�A
7�4
batch_normalization_41_input����������
p 

 
� "%�"
�
0���������
� �
J__inference_sequential_41_layer_call_and_return_conditional_losses_6940426�!"'(N�K
D�A
7�4
batch_normalization_41_input����������
p

 
� "%�"
�
0���������
� �
J__inference_sequential_41_layer_call_and_return_conditional_losses_6940567o!"'(8�5
.�+
!�
inputs����������
p 

 
� "%�"
�
0���������
� �
J__inference_sequential_41_layer_call_and_return_conditional_losses_6940627o!"'(8�5
.�+
!�
inputs����������
p

 
� "%�"
�
0���������
� �
/__inference_sequential_41_layer_call_fn_6940200x!"'(N�K
D�A
7�4
batch_normalization_41_input����������
p 

 
� "�����������
/__inference_sequential_41_layer_call_fn_6940360x!"'(N�K
D�A
7�4
batch_normalization_41_input����������
p

 
� "�����������
/__inference_sequential_41_layer_call_fn_6940492b!"'(8�5
.�+
!�
inputs����������
p 

 
� "�����������
/__inference_sequential_41_layer_call_fn_6940521b!"'(8�5
.�+
!�
inputs����������
p

 
� "�����������
%__inference_signature_wrapper_6940463�!"'(f�c
� 
\�Y
W
batch_normalization_41_input7�4
batch_normalization_41_input����������"5�2
0
	dense_167#� 
	dense_167���������