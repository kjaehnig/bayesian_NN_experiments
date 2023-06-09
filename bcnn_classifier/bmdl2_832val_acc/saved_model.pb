��<
�!�!
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
�
ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"!
output_typetype0	:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
.
Expm1
x"T
y"T"
Ttype:

2
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
�
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint���������"	
Ttype"
TItype0	:
2	
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
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
0
Sigmoid
x"T
y"T"
Ttype:

2
@
Softplus
features"T
activations"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
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
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��7
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
T
Const_1Const*
_output_shapes
:*
dtype0*
valueB*    
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  �?
^
Const_3Const*
_output_shapes
:	�*
dtype0*
valueB	�*    
T
Const_4Const*
_output_shapes
: *
dtype0*
valueB *    
L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *  �?
l
Const_6Const*&
_output_shapes
: *
dtype0*%
valueB *    
L
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *  �?
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
�
DAdam/v/dense_reparameterization_2/bias_posterior_untransformed_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDAdam/v/dense_reparameterization_2/bias_posterior_untransformed_scale
�
XAdam/v/dense_reparameterization_2/bias_posterior_untransformed_scale/Read/ReadVariableOpReadVariableOpDAdam/v/dense_reparameterization_2/bias_posterior_untransformed_scale*
_output_shapes
:*
dtype0
�
DAdam/m/dense_reparameterization_2/bias_posterior_untransformed_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDAdam/m/dense_reparameterization_2/bias_posterior_untransformed_scale
�
XAdam/m/dense_reparameterization_2/bias_posterior_untransformed_scale/Read/ReadVariableOpReadVariableOpDAdam/m/dense_reparameterization_2/bias_posterior_untransformed_scale*
_output_shapes
:*
dtype0
�
4Adam/v/dense_reparameterization_2/bias_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64Adam/v/dense_reparameterization_2/bias_posterior_loc
�
HAdam/v/dense_reparameterization_2/bias_posterior_loc/Read/ReadVariableOpReadVariableOp4Adam/v/dense_reparameterization_2/bias_posterior_loc*
_output_shapes
:*
dtype0
�
4Adam/m/dense_reparameterization_2/bias_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64Adam/m/dense_reparameterization_2/bias_posterior_loc
�
HAdam/m/dense_reparameterization_2/bias_posterior_loc/Read/ReadVariableOpReadVariableOp4Adam/m/dense_reparameterization_2/bias_posterior_loc*
_output_shapes
:*
dtype0
�
FAdam/v/dense_reparameterization_2/kernel_posterior_untransformed_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*W
shared_nameHFAdam/v/dense_reparameterization_2/kernel_posterior_untransformed_scale
�
ZAdam/v/dense_reparameterization_2/kernel_posterior_untransformed_scale/Read/ReadVariableOpReadVariableOpFAdam/v/dense_reparameterization_2/kernel_posterior_untransformed_scale*
_output_shapes
:	�*
dtype0
�
FAdam/m/dense_reparameterization_2/kernel_posterior_untransformed_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*W
shared_nameHFAdam/m/dense_reparameterization_2/kernel_posterior_untransformed_scale
�
ZAdam/m/dense_reparameterization_2/kernel_posterior_untransformed_scale/Read/ReadVariableOpReadVariableOpFAdam/m/dense_reparameterization_2/kernel_posterior_untransformed_scale*
_output_shapes
:	�*
dtype0
�
6Adam/v/dense_reparameterization_2/kernel_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*G
shared_name86Adam/v/dense_reparameterization_2/kernel_posterior_loc
�
JAdam/v/dense_reparameterization_2/kernel_posterior_loc/Read/ReadVariableOpReadVariableOp6Adam/v/dense_reparameterization_2/kernel_posterior_loc*
_output_shapes
:	�*
dtype0
�
6Adam/m/dense_reparameterization_2/kernel_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*G
shared_name86Adam/m/dense_reparameterization_2/kernel_posterior_loc
�
JAdam/m/dense_reparameterization_2/kernel_posterior_loc/Read/ReadVariableOpReadVariableOp6Adam/m/dense_reparameterization_2/kernel_posterior_loc*
_output_shapes
:	�*
dtype0
�
Adam/v/conv2d_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/v/conv2d_37/bias
|
)Adam/v/conv2d_37/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_37/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/conv2d_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/m/conv2d_37/bias
|
)Adam/m/conv2d_37/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_37/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/conv2d_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/v/conv2d_37/kernel
�
+Adam/v/conv2d_37/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_37/kernel*(
_output_shapes
:��*
dtype0
�
Adam/m/conv2d_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/m/conv2d_37/kernel
�
+Adam/m/conv2d_37/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_37/kernel*(
_output_shapes
:��*
dtype0
�
Adam/v/conv2d_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/v/conv2d_36/bias
|
)Adam/v/conv2d_36/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_36/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/conv2d_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/m/conv2d_36/bias
|
)Adam/m/conv2d_36/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_36/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/conv2d_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/v/conv2d_36/kernel
�
+Adam/v/conv2d_36/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_36/kernel*(
_output_shapes
:��*
dtype0
�
Adam/m/conv2d_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/m/conv2d_36/kernel
�
+Adam/m/conv2d_36/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_36/kernel*(
_output_shapes
:��*
dtype0
�
Adam/v/conv2d_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/v/conv2d_35/bias
|
)Adam/v/conv2d_35/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_35/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/conv2d_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/m/conv2d_35/bias
|
)Adam/m/conv2d_35/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_35/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/conv2d_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*(
shared_nameAdam/v/conv2d_35/kernel
�
+Adam/v/conv2d_35/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_35/kernel*'
_output_shapes
:@�*
dtype0
�
Adam/m/conv2d_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*(
shared_nameAdam/m/conv2d_35/kernel
�
+Adam/m/conv2d_35/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_35/kernel*'
_output_shapes
:@�*
dtype0
�
Adam/v/conv2d_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/v/conv2d_34/bias
{
)Adam/v/conv2d_34/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_34/bias*
_output_shapes
:@*
dtype0
�
Adam/m/conv2d_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/m/conv2d_34/bias
{
)Adam/m/conv2d_34/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_34/bias*
_output_shapes
:@*
dtype0
�
Adam/v/conv2d_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/v/conv2d_34/kernel
�
+Adam/v/conv2d_34/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_34/kernel*&
_output_shapes
:@@*
dtype0
�
Adam/m/conv2d_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/m/conv2d_34/kernel
�
+Adam/m/conv2d_34/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_34/kernel*&
_output_shapes
:@@*
dtype0
�
Adam/v/conv2d_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/v/conv2d_33/bias
{
)Adam/v/conv2d_33/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_33/bias*
_output_shapes
:@*
dtype0
�
Adam/m/conv2d_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/m/conv2d_33/bias
{
)Adam/m/conv2d_33/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_33/bias*
_output_shapes
:@*
dtype0
�
Adam/v/conv2d_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@*(
shared_nameAdam/v/conv2d_33/kernel
�
+Adam/v/conv2d_33/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_33/kernel*'
_output_shapes
:�@*
dtype0
�
Adam/m/conv2d_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@*(
shared_nameAdam/m/conv2d_33/kernel
�
+Adam/m/conv2d_33/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_33/kernel*'
_output_shapes
:�@*
dtype0
�
Adam/v/depthwise_conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!Adam/v/depthwise_conv2d_13/bias
�
3Adam/v/depthwise_conv2d_13/bias/Read/ReadVariableOpReadVariableOpAdam/v/depthwise_conv2d_13/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/depthwise_conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!Adam/m/depthwise_conv2d_13/bias
�
3Adam/m/depthwise_conv2d_13/bias/Read/ReadVariableOpReadVariableOpAdam/m/depthwise_conv2d_13/bias*
_output_shapes	
:�*
dtype0
�
+Adam/v/depthwise_conv2d_13/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/v/depthwise_conv2d_13/depthwise_kernel
�
?Adam/v/depthwise_conv2d_13/depthwise_kernel/Read/ReadVariableOpReadVariableOp+Adam/v/depthwise_conv2d_13/depthwise_kernel*&
_output_shapes
: *
dtype0
�
+Adam/m/depthwise_conv2d_13/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/m/depthwise_conv2d_13/depthwise_kernel
�
?Adam/m/depthwise_conv2d_13/depthwise_kernel/Read/ReadVariableOpReadVariableOp+Adam/m/depthwise_conv2d_13/depthwise_kernel*&
_output_shapes
: *
dtype0
�
EAdam/v/conv2d_reparameterization_2/bias_posterior_untransformed_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape: *V
shared_nameGEAdam/v/conv2d_reparameterization_2/bias_posterior_untransformed_scale
�
YAdam/v/conv2d_reparameterization_2/bias_posterior_untransformed_scale/Read/ReadVariableOpReadVariableOpEAdam/v/conv2d_reparameterization_2/bias_posterior_untransformed_scale*
_output_shapes
: *
dtype0
�
EAdam/m/conv2d_reparameterization_2/bias_posterior_untransformed_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape: *V
shared_nameGEAdam/m/conv2d_reparameterization_2/bias_posterior_untransformed_scale
�
YAdam/m/conv2d_reparameterization_2/bias_posterior_untransformed_scale/Read/ReadVariableOpReadVariableOpEAdam/m/conv2d_reparameterization_2/bias_posterior_untransformed_scale*
_output_shapes
: *
dtype0
�
5Adam/v/conv2d_reparameterization_2/bias_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape: *F
shared_name75Adam/v/conv2d_reparameterization_2/bias_posterior_loc
�
IAdam/v/conv2d_reparameterization_2/bias_posterior_loc/Read/ReadVariableOpReadVariableOp5Adam/v/conv2d_reparameterization_2/bias_posterior_loc*
_output_shapes
: *
dtype0
�
5Adam/m/conv2d_reparameterization_2/bias_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape: *F
shared_name75Adam/m/conv2d_reparameterization_2/bias_posterior_loc
�
IAdam/m/conv2d_reparameterization_2/bias_posterior_loc/Read/ReadVariableOpReadVariableOp5Adam/m/conv2d_reparameterization_2/bias_posterior_loc*
_output_shapes
: *
dtype0
�
GAdam/v/conv2d_reparameterization_2/kernel_posterior_untransformed_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/v/conv2d_reparameterization_2/kernel_posterior_untransformed_scale
�
[Adam/v/conv2d_reparameterization_2/kernel_posterior_untransformed_scale/Read/ReadVariableOpReadVariableOpGAdam/v/conv2d_reparameterization_2/kernel_posterior_untransformed_scale*&
_output_shapes
: *
dtype0
�
GAdam/m/conv2d_reparameterization_2/kernel_posterior_untransformed_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/m/conv2d_reparameterization_2/kernel_posterior_untransformed_scale
�
[Adam/m/conv2d_reparameterization_2/kernel_posterior_untransformed_scale/Read/ReadVariableOpReadVariableOpGAdam/m/conv2d_reparameterization_2/kernel_posterior_untransformed_scale*&
_output_shapes
: *
dtype0
�
7Adam/v/conv2d_reparameterization_2/kernel_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/v/conv2d_reparameterization_2/kernel_posterior_loc
�
KAdam/v/conv2d_reparameterization_2/kernel_posterior_loc/Read/ReadVariableOpReadVariableOp7Adam/v/conv2d_reparameterization_2/kernel_posterior_loc*&
_output_shapes
: *
dtype0
�
7Adam/m/conv2d_reparameterization_2/kernel_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/m/conv2d_reparameterization_2/kernel_posterior_loc
�
KAdam/m/conv2d_reparameterization_2/kernel_posterior_loc/Read/ReadVariableOpReadVariableOp7Adam/m/conv2d_reparameterization_2/kernel_posterior_loc*&
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
=dense_reparameterization_2/bias_posterior_untransformed_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=dense_reparameterization_2/bias_posterior_untransformed_scale
�
Qdense_reparameterization_2/bias_posterior_untransformed_scale/Read/ReadVariableOpReadVariableOp=dense_reparameterization_2/bias_posterior_untransformed_scale*
_output_shapes
:*
dtype0
�
-dense_reparameterization_2/bias_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-dense_reparameterization_2/bias_posterior_loc
�
Adense_reparameterization_2/bias_posterior_loc/Read/ReadVariableOpReadVariableOp-dense_reparameterization_2/bias_posterior_loc*
_output_shapes
:*
dtype0
�
?dense_reparameterization_2/kernel_posterior_untransformed_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*P
shared_nameA?dense_reparameterization_2/kernel_posterior_untransformed_scale
�
Sdense_reparameterization_2/kernel_posterior_untransformed_scale/Read/ReadVariableOpReadVariableOp?dense_reparameterization_2/kernel_posterior_untransformed_scale*
_output_shapes
:	�*
dtype0
�
/dense_reparameterization_2/kernel_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*@
shared_name1/dense_reparameterization_2/kernel_posterior_loc
�
Cdense_reparameterization_2/kernel_posterior_loc/Read/ReadVariableOpReadVariableOp/dense_reparameterization_2/kernel_posterior_loc*
_output_shapes
:	�*
dtype0
u
conv2d_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_37/bias
n
"conv2d_37/bias/Read/ReadVariableOpReadVariableOpconv2d_37/bias*
_output_shapes	
:�*
dtype0
�
conv2d_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_37/kernel

$conv2d_37/kernel/Read/ReadVariableOpReadVariableOpconv2d_37/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_36/bias
n
"conv2d_36/bias/Read/ReadVariableOpReadVariableOpconv2d_36/bias*
_output_shapes	
:�*
dtype0
�
conv2d_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_36/kernel

$conv2d_36/kernel/Read/ReadVariableOpReadVariableOpconv2d_36/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_35/bias
n
"conv2d_35/bias/Read/ReadVariableOpReadVariableOpconv2d_35/bias*
_output_shapes	
:�*
dtype0
�
conv2d_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*!
shared_nameconv2d_35/kernel
~
$conv2d_35/kernel/Read/ReadVariableOpReadVariableOpconv2d_35/kernel*'
_output_shapes
:@�*
dtype0
t
conv2d_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_34/bias
m
"conv2d_34/bias/Read/ReadVariableOpReadVariableOpconv2d_34/bias*
_output_shapes
:@*
dtype0
�
conv2d_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_34/kernel
}
$conv2d_34/kernel/Read/ReadVariableOpReadVariableOpconv2d_34/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_33/bias
m
"conv2d_33/bias/Read/ReadVariableOpReadVariableOpconv2d_33/bias*
_output_shapes
:@*
dtype0
�
conv2d_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@*!
shared_nameconv2d_33/kernel
~
$conv2d_33/kernel/Read/ReadVariableOpReadVariableOpconv2d_33/kernel*'
_output_shapes
:�@*
dtype0
�
depthwise_conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_namedepthwise_conv2d_13/bias
�
,depthwise_conv2d_13/bias/Read/ReadVariableOpReadVariableOpdepthwise_conv2d_13/bias*
_output_shapes	
:�*
dtype0
�
$depthwise_conv2d_13/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$depthwise_conv2d_13/depthwise_kernel
�
8depthwise_conv2d_13/depthwise_kernel/Read/ReadVariableOpReadVariableOp$depthwise_conv2d_13/depthwise_kernel*&
_output_shapes
: *
dtype0
�
>conv2d_reparameterization_2/bias_posterior_untransformed_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>conv2d_reparameterization_2/bias_posterior_untransformed_scale
�
Rconv2d_reparameterization_2/bias_posterior_untransformed_scale/Read/ReadVariableOpReadVariableOp>conv2d_reparameterization_2/bias_posterior_untransformed_scale*
_output_shapes
: *
dtype0
�
.conv2d_reparameterization_2/bias_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.conv2d_reparameterization_2/bias_posterior_loc
�
Bconv2d_reparameterization_2/bias_posterior_loc/Read/ReadVariableOpReadVariableOp.conv2d_reparameterization_2/bias_posterior_loc*
_output_shapes
: *
dtype0
�
@conv2d_reparameterization_2/kernel_posterior_untransformed_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@conv2d_reparameterization_2/kernel_posterior_untransformed_scale
�
Tconv2d_reparameterization_2/kernel_posterior_untransformed_scale/Read/ReadVariableOpReadVariableOp@conv2d_reparameterization_2/kernel_posterior_untransformed_scale*&
_output_shapes
: *
dtype0
�
0conv2d_reparameterization_2/kernel_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20conv2d_reparameterization_2/kernel_posterior_loc
�
Dconv2d_reparameterization_2/kernel_posterior_loc/Read/ReadVariableOpReadVariableOp0conv2d_reparameterization_2/kernel_posterior_loc*&
_output_shapes
: *
dtype0
�
1serving_default_conv2d_reparameterization_2_inputPlaceholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCall1serving_default_conv2d_reparameterization_2_input0conv2d_reparameterization_2/kernel_posterior_loc@conv2d_reparameterization_2/kernel_posterior_untransformed_scale.conv2d_reparameterization_2/bias_posterior_loc>conv2d_reparameterization_2/bias_posterior_untransformed_scaleConst_7Const_6Const_5Const_4$depthwise_conv2d_13/depthwise_kerneldepthwise_conv2d_13/biasconv2d_33/kernelconv2d_33/biasconv2d_34/kernelconv2d_34/biasconv2d_35/kernelconv2d_35/biasconv2d_36/kernelconv2d_36/biasconv2d_37/kernelconv2d_37/bias/dense_reparameterization_2/kernel_posterior_loc?dense_reparameterization_2/kernel_posterior_untransformed_scale-dense_reparameterization_2/bias_posterior_loc=dense_reparameterization_2/bias_posterior_untransformed_scaleConstConst_3Const_2Const_1*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_2149182

NoOpNoOp
��
Const_8Const"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer-12
layer-13
layer_with_weights-5
layer-14
layer_with_weights-6
layer-15
layer-16
layer-17
layer-18
layer_with_weights-7
layer-19
layer-20
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%kernel_posterior_loc
(&$kernel_posterior_untransformed_scale
'kernel_posterior
(kernel_prior
)bias_posterior_loc
&*"bias_posterior_untransformed_scale
+bias_posterior
,
bias_prior*
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses* 
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
9
activation
:depthwise_kernel
;bias
 <_jit_compiled_convolution_op*
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses* 
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
I_random_generator* 
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
9
activation

Pkernel
Qbias
 R_jit_compiled_convolution_op*
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses* 
�
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses
__random_generator* 
�
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses
9
activation

fkernel
gbias
 h_jit_compiled_convolution_op*
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses* 
�
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses
u_random_generator* 
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses
9
activation

|kernel
}bias
 ~_jit_compiled_convolution_op*
�
	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
9
activation
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
9
activation
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel_posterior_loc
)�$kernel_posterior_untransformed_scale
�kernel_posterior
�kernel_prior
�bias_posterior_loc
'�"bias_posterior_untransformed_scale
�bias_posterior
�
bias_prior*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_kwargs* 
�
%0
&1
)2
*3
:4
;5
P6
Q7
f8
g9
|10
}11
�12
�13
�14
�15
�16
�17
�18
�19*
�
%0
&1
)2
*3
:4
;5
P6
Q7
f8
g9
|10
}11
�12
�13
�14
�15
�16
�17
�18
�19*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
�
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�
capture_24
�
capture_25
�
capture_26
�
capture_27* 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla*

�serving_default* 
 
%0
&1
)2
*3*
 
%0
&1
)2
*3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
��
VARIABLE_VALUE0conv2d_reparameterization_2/kernel_posterior_locDlayer_with_weights-0/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE@conv2d_reparameterization_2/kernel_posterior_untransformed_scaleTlayer_with_weights-0/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUE*
-
�_distribution
�_graph_parents*
+
�_distribution
�_graph_parents* 
��
VARIABLE_VALUE.conv2d_reparameterization_2/bias_posterior_locBlayer_with_weights-0/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE>conv2d_reparameterization_2/bias_posterior_untransformed_scaleRlayer_with_weights-0/bias_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUE*
-
�_distribution
�_graph_parents*
+
�_distribution
�_graph_parents* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

:0
;1*

:0
;1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
~x
VARIABLE_VALUE$depthwise_conv2d_13/depthwise_kernel@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEdepthwise_conv2d_13/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

P0
Q1*

P0
Q1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_33/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_33/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

f0
g1*

f0
g1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_34/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_34/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

|0
}1*

|0
}1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_35/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_35/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_36/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_36/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_37/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_37/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
$
�0
�1
�2
�3*
$
�0
�1
�2
�3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
��
VARIABLE_VALUE/dense_reparameterization_2/kernel_posterior_locDlayer_with_weights-7/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE?dense_reparameterization_2/kernel_posterior_untransformed_scaleTlayer_with_weights-7/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUE*
-
�_distribution
�_graph_parents*
+
�_distribution
�_graph_parents* 
��
VARIABLE_VALUE-dense_reparameterization_2/bias_posterior_locBlayer_with_weights-7/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE=dense_reparameterization_2/bias_posterior_untransformed_scaleRlayer_with_weights-7/bias_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUE*
-
�_distribution
�_graph_parents*
+
�_distribution
�_graph_parents* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20*

�0
�1*
* 
* 
�
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�
capture_24
�
capture_25
�
capture_26
�
capture_27* 
�
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�
capture_24
�
capture_25
�
capture_26
�
capture_27* 
�
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�
capture_24
�
capture_25
�
capture_26
�
capture_27* 
�
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�
capture_24
�
capture_25
�
capture_26
�
capture_27* 
�
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�
capture_24
�
capture_25
�
capture_26
�
capture_27* 
�
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�
capture_24
�
capture_25
�
capture_26
�
capture_27* 
�
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�
capture_24
�
capture_25
�
capture_26
�
capture_27* 
�
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�
capture_24
�
capture_25
�
capture_26
�
capture_27* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19*
* 
�
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�
capture_24
�
capture_25
�
capture_26
�
capture_27* 
* 
* 
* 
* 
* 
B
�	capture_4
�	capture_5
�	capture_6
�	capture_7* 
B
�	capture_4
�	capture_5
�	capture_6
�	capture_7* 
0
%_loc
�_scale
�_graph_parents*
* 

�_graph_parents* 
* 
0
)_loc
�_scale
�_graph_parents*
* 

�_graph_parents* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
90* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
90* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
90* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
90* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
90* 
* 
* 
* 
* 
* 
* 
	
90* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
B
�	capture_4
�	capture_5
�	capture_6
�	capture_7* 
B
�	capture_4
�	capture_5
�	capture_6
�	capture_7* 
1
	�_loc
�_scale
�_graph_parents*
* 

�_graph_parents* 
* 
1
	�_loc
�_scale
�_graph_parents*
* 

�_graph_parents* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
�|
VARIABLE_VALUE7Adam/m/conv2d_reparameterization_2/kernel_posterior_loc1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUE7Adam/v/conv2d_reparameterization_2/kernel_posterior_loc1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEGAdam/m/conv2d_reparameterization_2/kernel_posterior_untransformed_scale1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEGAdam/v/conv2d_reparameterization_2/kernel_posterior_untransformed_scale1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE5Adam/m/conv2d_reparameterization_2/bias_posterior_loc1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE5Adam/v/conv2d_reparameterization_2/bias_posterior_loc1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEEAdam/m/conv2d_reparameterization_2/bias_posterior_untransformed_scale1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEEAdam/v/conv2d_reparameterization_2/bias_posterior_untransformed_scale1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE+Adam/m/depthwise_conv2d_13/depthwise_kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE+Adam/v/depthwise_conv2d_13/depthwise_kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/m/depthwise_conv2d_13/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/v/depthwise_conv2d_13/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_33/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_33/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_33/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_33/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_34/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_34/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_34/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_34/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_35/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_35/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_35/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_35/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_36/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_36/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_36/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_36/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_37/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_37/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_37/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_37/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUE6Adam/m/dense_reparameterization_2/kernel_posterior_loc2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUE6Adam/v/dense_reparameterization_2/kernel_posterior_loc2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEFAdam/m/dense_reparameterization_2/kernel_posterior_untransformed_scale2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEFAdam/v/dense_reparameterization_2/kernel_posterior_untransformed_scale2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE4Adam/m/dense_reparameterization_2/bias_posterior_loc2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE4Adam/v/dense_reparameterization_2/bias_posterior_loc2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEDAdam/m/dense_reparameterization_2/bias_posterior_untransformed_scale2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEDAdam/v/dense_reparameterization_2/bias_posterior_untransformed_scale2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*

&_pretransformed_input*
* 
* 

*_pretransformed_input*
* 
* 
* 
* 
* 
* 
* 
 
�_pretransformed_input*
* 
* 
 
�_pretransformed_input*
* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0conv2d_reparameterization_2/kernel_posterior_loc@conv2d_reparameterization_2/kernel_posterior_untransformed_scale.conv2d_reparameterization_2/bias_posterior_loc>conv2d_reparameterization_2/bias_posterior_untransformed_scale$depthwise_conv2d_13/depthwise_kerneldepthwise_conv2d_13/biasconv2d_33/kernelconv2d_33/biasconv2d_34/kernelconv2d_34/biasconv2d_35/kernelconv2d_35/biasconv2d_36/kernelconv2d_36/biasconv2d_37/kernelconv2d_37/bias/dense_reparameterization_2/kernel_posterior_loc?dense_reparameterization_2/kernel_posterior_untransformed_scale-dense_reparameterization_2/bias_posterior_loc=dense_reparameterization_2/bias_posterior_untransformed_scale	iterationlearning_rate7Adam/m/conv2d_reparameterization_2/kernel_posterior_loc7Adam/v/conv2d_reparameterization_2/kernel_posterior_locGAdam/m/conv2d_reparameterization_2/kernel_posterior_untransformed_scaleGAdam/v/conv2d_reparameterization_2/kernel_posterior_untransformed_scale5Adam/m/conv2d_reparameterization_2/bias_posterior_loc5Adam/v/conv2d_reparameterization_2/bias_posterior_locEAdam/m/conv2d_reparameterization_2/bias_posterior_untransformed_scaleEAdam/v/conv2d_reparameterization_2/bias_posterior_untransformed_scale+Adam/m/depthwise_conv2d_13/depthwise_kernel+Adam/v/depthwise_conv2d_13/depthwise_kernelAdam/m/depthwise_conv2d_13/biasAdam/v/depthwise_conv2d_13/biasAdam/m/conv2d_33/kernelAdam/v/conv2d_33/kernelAdam/m/conv2d_33/biasAdam/v/conv2d_33/biasAdam/m/conv2d_34/kernelAdam/v/conv2d_34/kernelAdam/m/conv2d_34/biasAdam/v/conv2d_34/biasAdam/m/conv2d_35/kernelAdam/v/conv2d_35/kernelAdam/m/conv2d_35/biasAdam/v/conv2d_35/biasAdam/m/conv2d_36/kernelAdam/v/conv2d_36/kernelAdam/m/conv2d_36/biasAdam/v/conv2d_36/biasAdam/m/conv2d_37/kernelAdam/v/conv2d_37/kernelAdam/m/conv2d_37/biasAdam/v/conv2d_37/bias6Adam/m/dense_reparameterization_2/kernel_posterior_loc6Adam/v/dense_reparameterization_2/kernel_posterior_locFAdam/m/dense_reparameterization_2/kernel_posterior_untransformed_scaleFAdam/v/dense_reparameterization_2/kernel_posterior_untransformed_scale4Adam/m/dense_reparameterization_2/bias_posterior_loc4Adam/v/dense_reparameterization_2/bias_posterior_locDAdam/m/dense_reparameterization_2/bias_posterior_untransformed_scaleDAdam/v/dense_reparameterization_2/bias_posterior_untransformed_scaletotal_1count_1totalcountConst_8*O
TinH
F2D*
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
 __inference__traced_save_2151189
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename0conv2d_reparameterization_2/kernel_posterior_loc@conv2d_reparameterization_2/kernel_posterior_untransformed_scale.conv2d_reparameterization_2/bias_posterior_loc>conv2d_reparameterization_2/bias_posterior_untransformed_scale$depthwise_conv2d_13/depthwise_kerneldepthwise_conv2d_13/biasconv2d_33/kernelconv2d_33/biasconv2d_34/kernelconv2d_34/biasconv2d_35/kernelconv2d_35/biasconv2d_36/kernelconv2d_36/biasconv2d_37/kernelconv2d_37/bias/dense_reparameterization_2/kernel_posterior_loc?dense_reparameterization_2/kernel_posterior_untransformed_scale-dense_reparameterization_2/bias_posterior_loc=dense_reparameterization_2/bias_posterior_untransformed_scale	iterationlearning_rate7Adam/m/conv2d_reparameterization_2/kernel_posterior_loc7Adam/v/conv2d_reparameterization_2/kernel_posterior_locGAdam/m/conv2d_reparameterization_2/kernel_posterior_untransformed_scaleGAdam/v/conv2d_reparameterization_2/kernel_posterior_untransformed_scale5Adam/m/conv2d_reparameterization_2/bias_posterior_loc5Adam/v/conv2d_reparameterization_2/bias_posterior_locEAdam/m/conv2d_reparameterization_2/bias_posterior_untransformed_scaleEAdam/v/conv2d_reparameterization_2/bias_posterior_untransformed_scale+Adam/m/depthwise_conv2d_13/depthwise_kernel+Adam/v/depthwise_conv2d_13/depthwise_kernelAdam/m/depthwise_conv2d_13/biasAdam/v/depthwise_conv2d_13/biasAdam/m/conv2d_33/kernelAdam/v/conv2d_33/kernelAdam/m/conv2d_33/biasAdam/v/conv2d_33/biasAdam/m/conv2d_34/kernelAdam/v/conv2d_34/kernelAdam/m/conv2d_34/biasAdam/v/conv2d_34/biasAdam/m/conv2d_35/kernelAdam/v/conv2d_35/kernelAdam/m/conv2d_35/biasAdam/v/conv2d_35/biasAdam/m/conv2d_36/kernelAdam/v/conv2d_36/kernelAdam/m/conv2d_36/biasAdam/v/conv2d_36/biasAdam/m/conv2d_37/kernelAdam/v/conv2d_37/kernelAdam/m/conv2d_37/biasAdam/v/conv2d_37/bias6Adam/m/dense_reparameterization_2/kernel_posterior_loc6Adam/v/dense_reparameterization_2/kernel_posterior_locFAdam/m/dense_reparameterization_2/kernel_posterior_untransformed_scaleFAdam/v/dense_reparameterization_2/kernel_posterior_untransformed_scale4Adam/m/dense_reparameterization_2/bias_posterior_loc4Adam/v/dense_reparameterization_2/bias_posterior_locDAdam/m/dense_reparameterization_2/bias_posterior_untransformed_scaleDAdam/v/dense_reparameterization_2/bias_posterior_untransformed_scaletotal_1count_1totalcount*N
TinG
E2C*
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
#__inference__traced_restore_2151397��3
�
H
,__inference_dropout_38_layer_call_fn_2150394

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_38_layer_call_and_return_conditional_losses_2148422h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������		@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������		@:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
i
M__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_2147764

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_37_layer_call_and_return_conditional_losses_2148152

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
activation_16/SigmoidSigmoidBiasAdd:output:0*
T0*0
_output_shapes
:�����������
activation_16/mulMulactivation_16/Sigmoid:y:0BiasAdd:output:0*
T0*0
_output_shapes
:����������m
IdentityIdentityactivation_16/mul:z:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
��
�E
 __inference__traced_save_2151189
file_prefixa
Gread_disablecopyonread_conv2d_reparameterization_2_kernel_posterior_loc: s
Yread_1_disablecopyonread_conv2d_reparameterization_2_kernel_posterior_untransformed_scale: U
Gread_2_disablecopyonread_conv2d_reparameterization_2_bias_posterior_loc: e
Wread_3_disablecopyonread_conv2d_reparameterization_2_bias_posterior_untransformed_scale: W
=read_4_disablecopyonread_depthwise_conv2d_13_depthwise_kernel: @
1read_5_disablecopyonread_depthwise_conv2d_13_bias:	�D
)read_6_disablecopyonread_conv2d_33_kernel:�@5
'read_7_disablecopyonread_conv2d_33_bias:@C
)read_8_disablecopyonread_conv2d_34_kernel:@@5
'read_9_disablecopyonread_conv2d_34_bias:@E
*read_10_disablecopyonread_conv2d_35_kernel:@�7
(read_11_disablecopyonread_conv2d_35_bias:	�F
*read_12_disablecopyonread_conv2d_36_kernel:��7
(read_13_disablecopyonread_conv2d_36_bias:	�F
*read_14_disablecopyonread_conv2d_37_kernel:��7
(read_15_disablecopyonread_conv2d_37_bias:	�\
Iread_16_disablecopyonread_dense_reparameterization_2_kernel_posterior_loc:	�l
Yread_17_disablecopyonread_dense_reparameterization_2_kernel_posterior_untransformed_scale:	�U
Gread_18_disablecopyonread_dense_reparameterization_2_bias_posterior_loc:e
Wread_19_disablecopyonread_dense_reparameterization_2_bias_posterior_untransformed_scale:-
#read_20_disablecopyonread_iteration:	 1
'read_21_disablecopyonread_learning_rate: k
Qread_22_disablecopyonread_adam_m_conv2d_reparameterization_2_kernel_posterior_loc: k
Qread_23_disablecopyonread_adam_v_conv2d_reparameterization_2_kernel_posterior_loc: {
aread_24_disablecopyonread_adam_m_conv2d_reparameterization_2_kernel_posterior_untransformed_scale: {
aread_25_disablecopyonread_adam_v_conv2d_reparameterization_2_kernel_posterior_untransformed_scale: ]
Oread_26_disablecopyonread_adam_m_conv2d_reparameterization_2_bias_posterior_loc: ]
Oread_27_disablecopyonread_adam_v_conv2d_reparameterization_2_bias_posterior_loc: m
_read_28_disablecopyonread_adam_m_conv2d_reparameterization_2_bias_posterior_untransformed_scale: m
_read_29_disablecopyonread_adam_v_conv2d_reparameterization_2_bias_posterior_untransformed_scale: _
Eread_30_disablecopyonread_adam_m_depthwise_conv2d_13_depthwise_kernel: _
Eread_31_disablecopyonread_adam_v_depthwise_conv2d_13_depthwise_kernel: H
9read_32_disablecopyonread_adam_m_depthwise_conv2d_13_bias:	�H
9read_33_disablecopyonread_adam_v_depthwise_conv2d_13_bias:	�L
1read_34_disablecopyonread_adam_m_conv2d_33_kernel:�@L
1read_35_disablecopyonread_adam_v_conv2d_33_kernel:�@=
/read_36_disablecopyonread_adam_m_conv2d_33_bias:@=
/read_37_disablecopyonread_adam_v_conv2d_33_bias:@K
1read_38_disablecopyonread_adam_m_conv2d_34_kernel:@@K
1read_39_disablecopyonread_adam_v_conv2d_34_kernel:@@=
/read_40_disablecopyonread_adam_m_conv2d_34_bias:@=
/read_41_disablecopyonread_adam_v_conv2d_34_bias:@L
1read_42_disablecopyonread_adam_m_conv2d_35_kernel:@�L
1read_43_disablecopyonread_adam_v_conv2d_35_kernel:@�>
/read_44_disablecopyonread_adam_m_conv2d_35_bias:	�>
/read_45_disablecopyonread_adam_v_conv2d_35_bias:	�M
1read_46_disablecopyonread_adam_m_conv2d_36_kernel:��M
1read_47_disablecopyonread_adam_v_conv2d_36_kernel:��>
/read_48_disablecopyonread_adam_m_conv2d_36_bias:	�>
/read_49_disablecopyonread_adam_v_conv2d_36_bias:	�M
1read_50_disablecopyonread_adam_m_conv2d_37_kernel:��M
1read_51_disablecopyonread_adam_v_conv2d_37_kernel:��>
/read_52_disablecopyonread_adam_m_conv2d_37_bias:	�>
/read_53_disablecopyonread_adam_v_conv2d_37_bias:	�c
Pread_54_disablecopyonread_adam_m_dense_reparameterization_2_kernel_posterior_loc:	�c
Pread_55_disablecopyonread_adam_v_dense_reparameterization_2_kernel_posterior_loc:	�s
`read_56_disablecopyonread_adam_m_dense_reparameterization_2_kernel_posterior_untransformed_scale:	�s
`read_57_disablecopyonread_adam_v_dense_reparameterization_2_kernel_posterior_untransformed_scale:	�\
Nread_58_disablecopyonread_adam_m_dense_reparameterization_2_bias_posterior_loc:\
Nread_59_disablecopyonread_adam_v_dense_reparameterization_2_bias_posterior_loc:l
^read_60_disablecopyonread_adam_m_dense_reparameterization_2_bias_posterior_untransformed_scale:l
^read_61_disablecopyonread_adam_v_dense_reparameterization_2_bias_posterior_untransformed_scale:+
!read_62_disablecopyonread_total_1: +
!read_63_disablecopyonread_count_1: )
read_64_disablecopyonread_total: )
read_65_disablecopyonread_count: 
savev2_const_8
identity_133��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
: �
Read/DisableCopyOnReadDisableCopyOnReadGread_disablecopyonread_conv2d_reparameterization_2_kernel_posterior_loc"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOpGread_disablecopyonread_conv2d_reparameterization_2_kernel_posterior_loc^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_1/DisableCopyOnReadDisableCopyOnReadYread_1_disablecopyonread_conv2d_reparameterization_2_kernel_posterior_untransformed_scale"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOpYread_1_disablecopyonread_conv2d_reparameterization_2_kernel_posterior_untransformed_scale^Read_1/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0u

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: k

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_2/DisableCopyOnReadDisableCopyOnReadGread_2_disablecopyonread_conv2d_reparameterization_2_bias_posterior_loc"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOpGread_2_disablecopyonread_conv2d_reparameterization_2_bias_posterior_loc^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_3/DisableCopyOnReadDisableCopyOnReadWread_3_disablecopyonread_conv2d_reparameterization_2_bias_posterior_untransformed_scale"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOpWread_3_disablecopyonread_conv2d_reparameterization_2_bias_posterior_untransformed_scale^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_4/DisableCopyOnReadDisableCopyOnRead=read_4_disablecopyonread_depthwise_conv2d_13_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp=read_4_disablecopyonread_depthwise_conv2d_13_depthwise_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0u

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_5/DisableCopyOnReadDisableCopyOnRead1read_5_disablecopyonread_depthwise_conv2d_13_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp1read_5_disablecopyonread_depthwise_conv2d_13_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_6/DisableCopyOnReadDisableCopyOnRead)read_6_disablecopyonread_conv2d_33_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp)read_6_disablecopyonread_conv2d_33_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:�@*
dtype0w
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:�@n
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*'
_output_shapes
:�@{
Read_7/DisableCopyOnReadDisableCopyOnRead'read_7_disablecopyonread_conv2d_33_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp'read_7_disablecopyonread_conv2d_33_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_8/DisableCopyOnReadDisableCopyOnRead)read_8_disablecopyonread_conv2d_34_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp)read_8_disablecopyonread_conv2d_34_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0v
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@{
Read_9/DisableCopyOnReadDisableCopyOnRead'read_9_disablecopyonread_conv2d_34_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp'read_9_disablecopyonread_conv2d_34_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_10/DisableCopyOnReadDisableCopyOnRead*read_10_disablecopyonread_conv2d_35_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp*read_10_disablecopyonread_conv2d_35_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0x
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*'
_output_shapes
:@�}
Read_11/DisableCopyOnReadDisableCopyOnRead(read_11_disablecopyonread_conv2d_35_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp(read_11_disablecopyonread_conv2d_35_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:�
Read_12/DisableCopyOnReadDisableCopyOnRead*read_12_disablecopyonread_conv2d_36_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp*read_12_disablecopyonread_conv2d_36_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*(
_output_shapes
:��}
Read_13/DisableCopyOnReadDisableCopyOnRead(read_13_disablecopyonread_conv2d_36_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp(read_13_disablecopyonread_conv2d_36_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:�
Read_14/DisableCopyOnReadDisableCopyOnRead*read_14_disablecopyonread_conv2d_37_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp*read_14_disablecopyonread_conv2d_37_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*(
_output_shapes
:��}
Read_15/DisableCopyOnReadDisableCopyOnRead(read_15_disablecopyonread_conv2d_37_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp(read_15_disablecopyonread_conv2d_37_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_16/DisableCopyOnReadDisableCopyOnReadIread_16_disablecopyonread_dense_reparameterization_2_kernel_posterior_loc"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOpIread_16_disablecopyonread_dense_reparameterization_2_kernel_posterior_loc^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_17/DisableCopyOnReadDisableCopyOnReadYread_17_disablecopyonread_dense_reparameterization_2_kernel_posterior_untransformed_scale"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOpYread_17_disablecopyonread_dense_reparameterization_2_kernel_posterior_untransformed_scale^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_18/DisableCopyOnReadDisableCopyOnReadGread_18_disablecopyonread_dense_reparameterization_2_bias_posterior_loc"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOpGread_18_disablecopyonread_dense_reparameterization_2_bias_posterior_loc^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_19/DisableCopyOnReadDisableCopyOnReadWread_19_disablecopyonread_dense_reparameterization_2_bias_posterior_untransformed_scale"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOpWread_19_disablecopyonread_dense_reparameterization_2_bias_posterior_untransformed_scale^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_20/DisableCopyOnReadDisableCopyOnRead#read_20_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp#read_20_disablecopyonread_iteration^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_21/DisableCopyOnReadDisableCopyOnRead'read_21_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp'read_21_disablecopyonread_learning_rate^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_22/DisableCopyOnReadDisableCopyOnReadQread_22_disablecopyonread_adam_m_conv2d_reparameterization_2_kernel_posterior_loc"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOpQread_22_disablecopyonread_adam_m_conv2d_reparameterization_2_kernel_posterior_loc^Read_22/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_23/DisableCopyOnReadDisableCopyOnReadQread_23_disablecopyonread_adam_v_conv2d_reparameterization_2_kernel_posterior_loc"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOpQread_23_disablecopyonread_adam_v_conv2d_reparameterization_2_kernel_posterior_loc^Read_23/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_24/DisableCopyOnReadDisableCopyOnReadaread_24_disablecopyonread_adam_m_conv2d_reparameterization_2_kernel_posterior_untransformed_scale"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOparead_24_disablecopyonread_adam_m_conv2d_reparameterization_2_kernel_posterior_untransformed_scale^Read_24/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_25/DisableCopyOnReadDisableCopyOnReadaread_25_disablecopyonread_adam_v_conv2d_reparameterization_2_kernel_posterior_untransformed_scale"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOparead_25_disablecopyonread_adam_v_conv2d_reparameterization_2_kernel_posterior_untransformed_scale^Read_25/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_26/DisableCopyOnReadDisableCopyOnReadOread_26_disablecopyonread_adam_m_conv2d_reparameterization_2_bias_posterior_loc"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOpOread_26_disablecopyonread_adam_m_conv2d_reparameterization_2_bias_posterior_loc^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_27/DisableCopyOnReadDisableCopyOnReadOread_27_disablecopyonread_adam_v_conv2d_reparameterization_2_bias_posterior_loc"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOpOread_27_disablecopyonread_adam_v_conv2d_reparameterization_2_bias_posterior_loc^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_28/DisableCopyOnReadDisableCopyOnRead_read_28_disablecopyonread_adam_m_conv2d_reparameterization_2_bias_posterior_untransformed_scale"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp_read_28_disablecopyonread_adam_m_conv2d_reparameterization_2_bias_posterior_untransformed_scale^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_29/DisableCopyOnReadDisableCopyOnRead_read_29_disablecopyonread_adam_v_conv2d_reparameterization_2_bias_posterior_untransformed_scale"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp_read_29_disablecopyonread_adam_v_conv2d_reparameterization_2_bias_posterior_untransformed_scale^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_30/DisableCopyOnReadDisableCopyOnReadEread_30_disablecopyonread_adam_m_depthwise_conv2d_13_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOpEread_30_disablecopyonread_adam_m_depthwise_conv2d_13_depthwise_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_31/DisableCopyOnReadDisableCopyOnReadEread_31_disablecopyonread_adam_v_depthwise_conv2d_13_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOpEread_31_disablecopyonread_adam_v_depthwise_conv2d_13_depthwise_kernel^Read_31/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_32/DisableCopyOnReadDisableCopyOnRead9read_32_disablecopyonread_adam_m_depthwise_conv2d_13_bias"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp9read_32_disablecopyonread_adam_m_depthwise_conv2d_13_bias^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_33/DisableCopyOnReadDisableCopyOnRead9read_33_disablecopyonread_adam_v_depthwise_conv2d_13_bias"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp9read_33_disablecopyonread_adam_v_depthwise_conv2d_13_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_34/DisableCopyOnReadDisableCopyOnRead1read_34_disablecopyonread_adam_m_conv2d_33_kernel"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp1read_34_disablecopyonread_adam_m_conv2d_33_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:�@*
dtype0x
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:�@n
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*'
_output_shapes
:�@�
Read_35/DisableCopyOnReadDisableCopyOnRead1read_35_disablecopyonread_adam_v_conv2d_33_kernel"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp1read_35_disablecopyonread_adam_v_conv2d_33_kernel^Read_35/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:�@*
dtype0x
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:�@n
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*'
_output_shapes
:�@�
Read_36/DisableCopyOnReadDisableCopyOnRead/read_36_disablecopyonread_adam_m_conv2d_33_bias"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp/read_36_disablecopyonread_adam_m_conv2d_33_bias^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_37/DisableCopyOnReadDisableCopyOnRead/read_37_disablecopyonread_adam_v_conv2d_33_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp/read_37_disablecopyonread_adam_v_conv2d_33_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_38/DisableCopyOnReadDisableCopyOnRead1read_38_disablecopyonread_adam_m_conv2d_34_kernel"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp1read_38_disablecopyonread_adam_m_conv2d_34_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@�
Read_39/DisableCopyOnReadDisableCopyOnRead1read_39_disablecopyonread_adam_v_conv2d_34_kernel"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp1read_39_disablecopyonread_adam_v_conv2d_34_kernel^Read_39/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@�
Read_40/DisableCopyOnReadDisableCopyOnRead/read_40_disablecopyonread_adam_m_conv2d_34_bias"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp/read_40_disablecopyonread_adam_m_conv2d_34_bias^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_41/DisableCopyOnReadDisableCopyOnRead/read_41_disablecopyonread_adam_v_conv2d_34_bias"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp/read_41_disablecopyonread_adam_v_conv2d_34_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_42/DisableCopyOnReadDisableCopyOnRead1read_42_disablecopyonread_adam_m_conv2d_35_kernel"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp1read_42_disablecopyonread_adam_m_conv2d_35_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0x
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_43/DisableCopyOnReadDisableCopyOnRead1read_43_disablecopyonread_adam_v_conv2d_35_kernel"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp1read_43_disablecopyonread_adam_v_conv2d_35_kernel^Read_43/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0x
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_44/DisableCopyOnReadDisableCopyOnRead/read_44_disablecopyonread_adam_m_conv2d_35_bias"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp/read_44_disablecopyonread_adam_m_conv2d_35_bias^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_45/DisableCopyOnReadDisableCopyOnRead/read_45_disablecopyonread_adam_v_conv2d_35_bias"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp/read_45_disablecopyonread_adam_v_conv2d_35_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_46/DisableCopyOnReadDisableCopyOnRead1read_46_disablecopyonread_adam_m_conv2d_36_kernel"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp1read_46_disablecopyonread_adam_m_conv2d_36_kernel^Read_46/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_47/DisableCopyOnReadDisableCopyOnRead1read_47_disablecopyonread_adam_v_conv2d_36_kernel"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp1read_47_disablecopyonread_adam_v_conv2d_36_kernel^Read_47/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_48/DisableCopyOnReadDisableCopyOnRead/read_48_disablecopyonread_adam_m_conv2d_36_bias"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp/read_48_disablecopyonread_adam_m_conv2d_36_bias^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_49/DisableCopyOnReadDisableCopyOnRead/read_49_disablecopyonread_adam_v_conv2d_36_bias"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp/read_49_disablecopyonread_adam_v_conv2d_36_bias^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_50/DisableCopyOnReadDisableCopyOnRead1read_50_disablecopyonread_adam_m_conv2d_37_kernel"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp1read_50_disablecopyonread_adam_m_conv2d_37_kernel^Read_50/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_51/DisableCopyOnReadDisableCopyOnRead1read_51_disablecopyonread_adam_v_conv2d_37_kernel"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp1read_51_disablecopyonread_adam_v_conv2d_37_kernel^Read_51/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_52/DisableCopyOnReadDisableCopyOnRead/read_52_disablecopyonread_adam_m_conv2d_37_bias"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp/read_52_disablecopyonread_adam_m_conv2d_37_bias^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_53/DisableCopyOnReadDisableCopyOnRead/read_53_disablecopyonread_adam_v_conv2d_37_bias"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp/read_53_disablecopyonread_adam_v_conv2d_37_bias^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_54/DisableCopyOnReadDisableCopyOnReadPread_54_disablecopyonread_adam_m_dense_reparameterization_2_kernel_posterior_loc"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOpPread_54_disablecopyonread_adam_m_dense_reparameterization_2_kernel_posterior_loc^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0q
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_55/DisableCopyOnReadDisableCopyOnReadPread_55_disablecopyonread_adam_v_dense_reparameterization_2_kernel_posterior_loc"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOpPread_55_disablecopyonread_adam_v_dense_reparameterization_2_kernel_posterior_loc^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0q
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_56/DisableCopyOnReadDisableCopyOnRead`read_56_disablecopyonread_adam_m_dense_reparameterization_2_kernel_posterior_untransformed_scale"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp`read_56_disablecopyonread_adam_m_dense_reparameterization_2_kernel_posterior_untransformed_scale^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0q
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_57/DisableCopyOnReadDisableCopyOnRead`read_57_disablecopyonread_adam_v_dense_reparameterization_2_kernel_posterior_untransformed_scale"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp`read_57_disablecopyonread_adam_v_dense_reparameterization_2_kernel_posterior_untransformed_scale^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0q
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_58/DisableCopyOnReadDisableCopyOnReadNread_58_disablecopyonread_adam_m_dense_reparameterization_2_bias_posterior_loc"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOpNread_58_disablecopyonread_adam_m_dense_reparameterization_2_bias_posterior_loc^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_59/DisableCopyOnReadDisableCopyOnReadNread_59_disablecopyonread_adam_v_dense_reparameterization_2_bias_posterior_loc"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOpNread_59_disablecopyonread_adam_v_dense_reparameterization_2_bias_posterior_loc^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_60/DisableCopyOnReadDisableCopyOnRead^read_60_disablecopyonread_adam_m_dense_reparameterization_2_bias_posterior_untransformed_scale"/device:CPU:0*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOp^read_60_disablecopyonread_adam_m_dense_reparameterization_2_bias_posterior_untransformed_scale^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_61/DisableCopyOnReadDisableCopyOnRead^read_61_disablecopyonread_adam_v_dense_reparameterization_2_bias_posterior_untransformed_scale"/device:CPU:0*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOp^read_61_disablecopyonread_adam_v_dense_reparameterization_2_bias_posterior_untransformed_scale^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_62/DisableCopyOnReadDisableCopyOnRead!read_62_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOp!read_62_disablecopyonread_total_1^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_63/DisableCopyOnReadDisableCopyOnRead!read_63_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOp!read_63_disablecopyonread_count_1^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_64/DisableCopyOnReadDisableCopyOnReadread_64_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOpread_64_disablecopyonread_total^Read_64/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_65/DisableCopyOnReadDisableCopyOnReadread_65_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOpread_65_disablecopyonread_count^Read_65/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*�
value�B�CBDlayer_with_weights-0/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-7/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-7/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-7/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/bias_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*�
value�B�CB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0savev2_const_8"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *Q
dtypesG
E2C	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_132Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_133IdentityIdentity_132:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "%
identity_133Identity_133:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
~
R__inference_one_hot_categorical_7_layer_call_and_return_conditional_losses_2148484

inputs
identity

identity_1|
6tensor_coercible/value/OneHotCategorical/mode/IdentityIdentityinputs*
T0*'
_output_shapes
:����������
>tensor_coercible/value/OneHotCategorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
����������
4tensor_coercible/value/OneHotCategorical/mode/ArgMaxArgMax?tensor_coercible/value/OneHotCategorical/mode/Identity:output:0Gtensor_coercible/value/OneHotCategorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:����������
>tensor_coercible/value/OneHotCategorical/mode/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
?tensor_coercible/value/OneHotCategorical/mode/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    }
;tensor_coercible/value/OneHotCategorical/mode/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :�
5tensor_coercible/value/OneHotCategorical/mode/one_hotOneHot=tensor_coercible/value/OneHotCategorical/mode/ArgMax:output:0Dtensor_coercible/value/OneHotCategorical/mode/one_hot/depth:output:0Gtensor_coercible/value/OneHotCategorical/mode/one_hot/on_value:output:0Htensor_coercible/value/OneHotCategorical/mode/one_hot/off_value:output:0*
T0*'
_output_shapes
:����������
IdentityIdentity>tensor_coercible/value/OneHotCategorical/mode/one_hot:output:0*
T0*'
_output_shapes
:����������

Identity_1Identity>tensor_coercible/value/OneHotCategorical/mode/one_hot:output:0*
T0*'
_output_shapes
:���������"!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
e
G__inference_dropout_39_layer_call_and_return_conditional_losses_2148434

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:����������d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
n
R__inference_one_hot_categorical_7_layer_call_and_return_conditional_losses_2150762

inputs
identity|
6tensor_coercible/value/OneHotCategorical/mode/IdentityIdentityinputs*
T0*'
_output_shapes
:����������
>tensor_coercible/value/OneHotCategorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
����������
4tensor_coercible/value/OneHotCategorical/mode/ArgMaxArgMax?tensor_coercible/value/OneHotCategorical/mode/Identity:output:0Gtensor_coercible/value/OneHotCategorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:����������
>tensor_coercible/value/OneHotCategorical/mode/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
?tensor_coercible/value/OneHotCategorical/mode/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    }
;tensor_coercible/value/OneHotCategorical/mode/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :�
5tensor_coercible/value/OneHotCategorical/mode/one_hotOneHot=tensor_coercible/value/OneHotCategorical/mode/ArgMax:output:0Dtensor_coercible/value/OneHotCategorical/mode/one_hot/depth:output:0Gtensor_coercible/value/OneHotCategorical/mode/one_hot/on_value:output:0Htensor_coercible/value/OneHotCategorical/mode/one_hot/off_value:output:0*
T0*'
_output_shapes
:����������
IdentityIdentity>tensor_coercible/value/OneHotCategorical/mode/one_hot:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
H
,__inference_dropout_36_layer_call_fn_2150278

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_36_layer_call_and_return_conditional_losses_2148398i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:���������%%�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������%%�:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
�
e
G__inference_dropout_40_layer_call_and_return_conditional_losses_2150548

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:����������d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

f
G__inference_dropout_38_layer_call_and_return_conditional_losses_2150406

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������		@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������		@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������		@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:���������		@i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:���������		@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������		@:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
c
7__inference_one_hot_categorical_7_layer_call_fn_2150740

inputs
identity

identity_1�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_one_hot_categorical_7_layer_call_and_return_conditional_losses_2148484`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:���������"!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/__inference_sequential_14_layer_call_fn_2149247

inputs!
unknown: #
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3
	unknown_4
	unknown_5
	unknown_6#
	unknown_7: 
	unknown_8:	�$
	unknown_9:�@

unknown_10:@$

unknown_11:@@

unknown_12:@%

unknown_13:@�

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�

unknown_19:	�

unknown_20:	�

unknown_21:

unknown_22:

unknown_23

unknown_24

unknown_25

unknown_26
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout	
2*
_collective_manager_ids
 */
_output_shapes
:���������: : : : *6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_14_layer_call_and_return_conditional_losses_2148585o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:�����������: : : : : : : : : : : : : : : : : : : : : : : : : :	�: :22
StatefulPartitionedCallStatefulPartitionedCall: 

_output_shapes
::

_output_shapes
: :%!

_output_shapes
:	�:

_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
/__inference_sequential_14_layer_call_fn_2148648%
!conv2d_reparameterization_2_input!
unknown: #
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3
	unknown_4
	unknown_5
	unknown_6#
	unknown_7: 
	unknown_8:	�$
	unknown_9:�@

unknown_10:@$

unknown_11:@@

unknown_12:@%

unknown_13:@�

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�

unknown_19:	�

unknown_20:	�

unknown_21:

unknown_22:

unknown_23

unknown_24

unknown_25

unknown_26
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall!conv2d_reparameterization_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout	
2*
_collective_manager_ids
 */
_output_shapes
:���������: : : : *6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_14_layer_call_and_return_conditional_losses_2148585o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:�����������: : : : : : : : : : : : : : : : : : : : : : : : : :	�: :22
StatefulPartitionedCallStatefulPartitionedCall: 

_output_shapes
::

_output_shapes
: :%!

_output_shapes
:	�:

_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :t p
1
_output_shapes
:�����������
;
_user_specified_name#!conv2d_reparameterization_2_input
�
�
%__inference_signature_wrapper_2149182%
!conv2d_reparameterization_2_input!
unknown: #
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3
	unknown_4
	unknown_5
	unknown_6#
	unknown_7: 
	unknown_8:	�$
	unknown_9:�@

unknown_10:@$

unknown_11:@@

unknown_12:@%

unknown_13:@�

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�

unknown_19:	�

unknown_20:	�

unknown_21:

unknown_22:

unknown_23

unknown_24

unknown_25

unknown_26
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall!conv2d_reparameterization_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_2147746o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:�����������: : : : : : : : : : : : : : : : : : : : : : : : : :	�: :22
StatefulPartitionedCallStatefulPartitionedCall: 

_output_shapes
::

_output_shapes
: :%!

_output_shapes
:	�:

_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :t p
1
_output_shapes
:�����������
;
_user_specified_name#!conv2d_reparameterization_2_input
�
�
P__inference_depthwise_conv2d_13_layer_call_and_return_conditional_losses_2148002

inputs;
!depthwise_readvariableop_resource: .
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�depthwise/ReadVariableOp�
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
: *
dtype0h
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             h
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAdddepthwise:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�m
activation_16/SigmoidSigmoidBiasAdd:output:0*
T0*0
_output_shapes
:���������KK��
activation_16/mulMulactivation_16/Sigmoid:y:0BiasAdd:output:0*
T0*0
_output_shapes
:���������KK�m
IdentityIdentityactivation_16/mul:z:0^NoOp*
T0*0
_output_shapes
:���������KK�z
NoOpNoOp^BiasAdd/ReadVariableOp^depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������KK : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:W S
/
_output_shapes
:���������KK 
 
_user_specified_nameinputs
�
i
M__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_2150521

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
��
�2
#__inference__traced_restore_2151397
file_prefix[
Aassignvariableop_conv2d_reparameterization_2_kernel_posterior_loc: m
Sassignvariableop_1_conv2d_reparameterization_2_kernel_posterior_untransformed_scale: O
Aassignvariableop_2_conv2d_reparameterization_2_bias_posterior_loc: _
Qassignvariableop_3_conv2d_reparameterization_2_bias_posterior_untransformed_scale: Q
7assignvariableop_4_depthwise_conv2d_13_depthwise_kernel: :
+assignvariableop_5_depthwise_conv2d_13_bias:	�>
#assignvariableop_6_conv2d_33_kernel:�@/
!assignvariableop_7_conv2d_33_bias:@=
#assignvariableop_8_conv2d_34_kernel:@@/
!assignvariableop_9_conv2d_34_bias:@?
$assignvariableop_10_conv2d_35_kernel:@�1
"assignvariableop_11_conv2d_35_bias:	�@
$assignvariableop_12_conv2d_36_kernel:��1
"assignvariableop_13_conv2d_36_bias:	�@
$assignvariableop_14_conv2d_37_kernel:��1
"assignvariableop_15_conv2d_37_bias:	�V
Cassignvariableop_16_dense_reparameterization_2_kernel_posterior_loc:	�f
Sassignvariableop_17_dense_reparameterization_2_kernel_posterior_untransformed_scale:	�O
Aassignvariableop_18_dense_reparameterization_2_bias_posterior_loc:_
Qassignvariableop_19_dense_reparameterization_2_bias_posterior_untransformed_scale:'
assignvariableop_20_iteration:	 +
!assignvariableop_21_learning_rate: e
Kassignvariableop_22_adam_m_conv2d_reparameterization_2_kernel_posterior_loc: e
Kassignvariableop_23_adam_v_conv2d_reparameterization_2_kernel_posterior_loc: u
[assignvariableop_24_adam_m_conv2d_reparameterization_2_kernel_posterior_untransformed_scale: u
[assignvariableop_25_adam_v_conv2d_reparameterization_2_kernel_posterior_untransformed_scale: W
Iassignvariableop_26_adam_m_conv2d_reparameterization_2_bias_posterior_loc: W
Iassignvariableop_27_adam_v_conv2d_reparameterization_2_bias_posterior_loc: g
Yassignvariableop_28_adam_m_conv2d_reparameterization_2_bias_posterior_untransformed_scale: g
Yassignvariableop_29_adam_v_conv2d_reparameterization_2_bias_posterior_untransformed_scale: Y
?assignvariableop_30_adam_m_depthwise_conv2d_13_depthwise_kernel: Y
?assignvariableop_31_adam_v_depthwise_conv2d_13_depthwise_kernel: B
3assignvariableop_32_adam_m_depthwise_conv2d_13_bias:	�B
3assignvariableop_33_adam_v_depthwise_conv2d_13_bias:	�F
+assignvariableop_34_adam_m_conv2d_33_kernel:�@F
+assignvariableop_35_adam_v_conv2d_33_kernel:�@7
)assignvariableop_36_adam_m_conv2d_33_bias:@7
)assignvariableop_37_adam_v_conv2d_33_bias:@E
+assignvariableop_38_adam_m_conv2d_34_kernel:@@E
+assignvariableop_39_adam_v_conv2d_34_kernel:@@7
)assignvariableop_40_adam_m_conv2d_34_bias:@7
)assignvariableop_41_adam_v_conv2d_34_bias:@F
+assignvariableop_42_adam_m_conv2d_35_kernel:@�F
+assignvariableop_43_adam_v_conv2d_35_kernel:@�8
)assignvariableop_44_adam_m_conv2d_35_bias:	�8
)assignvariableop_45_adam_v_conv2d_35_bias:	�G
+assignvariableop_46_adam_m_conv2d_36_kernel:��G
+assignvariableop_47_adam_v_conv2d_36_kernel:��8
)assignvariableop_48_adam_m_conv2d_36_bias:	�8
)assignvariableop_49_adam_v_conv2d_36_bias:	�G
+assignvariableop_50_adam_m_conv2d_37_kernel:��G
+assignvariableop_51_adam_v_conv2d_37_kernel:��8
)assignvariableop_52_adam_m_conv2d_37_bias:	�8
)assignvariableop_53_adam_v_conv2d_37_bias:	�]
Jassignvariableop_54_adam_m_dense_reparameterization_2_kernel_posterior_loc:	�]
Jassignvariableop_55_adam_v_dense_reparameterization_2_kernel_posterior_loc:	�m
Zassignvariableop_56_adam_m_dense_reparameterization_2_kernel_posterior_untransformed_scale:	�m
Zassignvariableop_57_adam_v_dense_reparameterization_2_kernel_posterior_untransformed_scale:	�V
Hassignvariableop_58_adam_m_dense_reparameterization_2_bias_posterior_loc:V
Hassignvariableop_59_adam_v_dense_reparameterization_2_bias_posterior_loc:f
Xassignvariableop_60_adam_m_dense_reparameterization_2_bias_posterior_untransformed_scale:f
Xassignvariableop_61_adam_v_dense_reparameterization_2_bias_posterior_untransformed_scale:%
assignvariableop_62_total_1: %
assignvariableop_63_count_1: #
assignvariableop_64_total: #
assignvariableop_65_count: 
identity_67��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*�
value�B�CBDlayer_with_weights-0/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-7/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-7/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-7/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/bias_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*�
value�B�CB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Q
dtypesG
E2C	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpAassignvariableop_conv2d_reparameterization_2_kernel_posterior_locIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpSassignvariableop_1_conv2d_reparameterization_2_kernel_posterior_untransformed_scaleIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpAassignvariableop_2_conv2d_reparameterization_2_bias_posterior_locIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpQassignvariableop_3_conv2d_reparameterization_2_bias_posterior_untransformed_scaleIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp7assignvariableop_4_depthwise_conv2d_13_depthwise_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp+assignvariableop_5_depthwise_conv2d_13_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_33_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_33_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_34_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_34_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_35_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_35_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_36_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_36_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp$assignvariableop_14_conv2d_37_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp"assignvariableop_15_conv2d_37_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpCassignvariableop_16_dense_reparameterization_2_kernel_posterior_locIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpSassignvariableop_17_dense_reparameterization_2_kernel_posterior_untransformed_scaleIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpAassignvariableop_18_dense_reparameterization_2_bias_posterior_locIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpQassignvariableop_19_dense_reparameterization_2_bias_posterior_untransformed_scaleIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_iterationIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp!assignvariableop_21_learning_rateIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpKassignvariableop_22_adam_m_conv2d_reparameterization_2_kernel_posterior_locIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpKassignvariableop_23_adam_v_conv2d_reparameterization_2_kernel_posterior_locIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp[assignvariableop_24_adam_m_conv2d_reparameterization_2_kernel_posterior_untransformed_scaleIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp[assignvariableop_25_adam_v_conv2d_reparameterization_2_kernel_posterior_untransformed_scaleIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpIassignvariableop_26_adam_m_conv2d_reparameterization_2_bias_posterior_locIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpIassignvariableop_27_adam_v_conv2d_reparameterization_2_bias_posterior_locIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpYassignvariableop_28_adam_m_conv2d_reparameterization_2_bias_posterior_untransformed_scaleIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpYassignvariableop_29_adam_v_conv2d_reparameterization_2_bias_posterior_untransformed_scaleIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp?assignvariableop_30_adam_m_depthwise_conv2d_13_depthwise_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp?assignvariableop_31_adam_v_depthwise_conv2d_13_depthwise_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp3assignvariableop_32_adam_m_depthwise_conv2d_13_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp3assignvariableop_33_adam_v_depthwise_conv2d_13_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp+assignvariableop_34_adam_m_conv2d_33_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_v_conv2d_33_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_m_conv2d_33_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_v_conv2d_33_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp+assignvariableop_38_adam_m_conv2d_34_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_v_conv2d_34_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_m_conv2d_34_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_v_conv2d_34_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp+assignvariableop_42_adam_m_conv2d_35_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_v_conv2d_35_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_m_conv2d_35_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_v_conv2d_35_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp+assignvariableop_46_adam_m_conv2d_36_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_v_conv2d_36_kernelIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_m_conv2d_36_biasIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_v_conv2d_36_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp+assignvariableop_50_adam_m_conv2d_37_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_v_conv2d_37_kernelIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_m_conv2d_37_biasIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp)assignvariableop_53_adam_v_conv2d_37_biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOpJassignvariableop_54_adam_m_dense_reparameterization_2_kernel_posterior_locIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOpJassignvariableop_55_adam_v_dense_reparameterization_2_kernel_posterior_locIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOpZassignvariableop_56_adam_m_dense_reparameterization_2_kernel_posterior_untransformed_scaleIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOpZassignvariableop_57_adam_v_dense_reparameterization_2_kernel_posterior_untransformed_scaleIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOpHassignvariableop_58_adam_m_dense_reparameterization_2_bias_posterior_locIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOpHassignvariableop_59_adam_v_dense_reparameterization_2_bias_posterior_locIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOpXassignvariableop_60_adam_m_dense_reparameterization_2_bias_posterior_untransformed_scaleIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOpXassignvariableop_61_adam_v_dense_reparameterization_2_bias_posterior_untransformed_scaleIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOpassignvariableop_62_total_1Identity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOpassignvariableop_63_count_1Identity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOpassignvariableop_64_totalIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOpassignvariableop_65_countIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_66Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_67IdentityIdentity_66:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_67Identity_67:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
5__inference_depthwise_conv2d_13_layer_call_fn_2150244

inputs!
unknown: 
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������KK�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_depthwise_conv2d_13_layer_call_and_return_conditional_losses_2148002x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������KK�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������KK : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������KK 
 
_user_specified_nameinputs
�
�
F__inference_conv2d_34_layer_call_and_return_conditional_losses_2150374

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@l
activation_16/SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:���������@
activation_16/mulMulactivation_16/Sigmoid:y:0BiasAdd:output:0*
T0*/
_output_shapes
:���������@l
IdentityIdentityactivation_16/mul:z:0^NoOp*
T0*/
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�

f
G__inference_dropout_40_layer_call_and_return_conditional_losses_2150543

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentitydropout/SelectV2:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
i
M__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_2150326

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
=__inference_conv2d_reparameterization_2_layer_call_fn_2150080

inputs!
unknown: #
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:����������� : : *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *a
f\RZ
X__inference_conv2d_reparameterization_2_layer_call_and_return_conditional_losses_2147967y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:����������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:�����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
e
G__inference_dropout_38_layer_call_and_return_conditional_losses_2150411

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������		@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������		@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������		@:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
�
+__inference_conv2d_34_layer_call_fn_2150362

inputs!
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_34_layer_call_and_return_conditional_losses_2148068w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�n
�
J__inference_sequential_14_layer_call_and_return_conditional_losses_2148365%
!conv2d_reparameterization_2_input=
#conv2d_reparameterization_2_2147968: =
#conv2d_reparameterization_2_2147970: 1
#conv2d_reparameterization_2_2147972: 1
#conv2d_reparameterization_2_2147974: '
#conv2d_reparameterization_2_2147976'
#conv2d_reparameterization_2_2147978'
#conv2d_reparameterization_2_2147980'
#conv2d_reparameterization_2_21479825
depthwise_conv2d_13_2148003: *
depthwise_conv2d_13_2148005:	�,
conv2d_33_2148036:�@
conv2d_33_2148038:@+
conv2d_34_2148069:@@
conv2d_34_2148071:@,
conv2d_35_2148102:@� 
conv2d_35_2148104:	�-
conv2d_36_2148135:�� 
conv2d_36_2148137:	�-
conv2d_37_2148153:�� 
conv2d_37_2148155:	�5
"dense_reparameterization_2_2148326:	�5
"dense_reparameterization_2_2148328:	�0
"dense_reparameterization_2_2148330:0
"dense_reparameterization_2_2148332:&
"dense_reparameterization_2_2148334&
"dense_reparameterization_2_2148336&
"dense_reparameterization_2_2148338&
"dense_reparameterization_2_2148340
identity

identity_1

identity_2

identity_3

identity_4��!conv2d_33/StatefulPartitionedCall�!conv2d_34/StatefulPartitionedCall�!conv2d_35/StatefulPartitionedCall�!conv2d_36/StatefulPartitionedCall�!conv2d_37/StatefulPartitionedCall�3conv2d_reparameterization_2/StatefulPartitionedCall�2dense_reparameterization_2/StatefulPartitionedCall�+depthwise_conv2d_13/StatefulPartitionedCall�"dropout_36/StatefulPartitionedCall�"dropout_37/StatefulPartitionedCall�"dropout_38/StatefulPartitionedCall�"dropout_39/StatefulPartitionedCall�"dropout_40/StatefulPartitionedCall�
3conv2d_reparameterization_2/StatefulPartitionedCallStatefulPartitionedCall!conv2d_reparameterization_2_input#conv2d_reparameterization_2_2147968#conv2d_reparameterization_2_2147970#conv2d_reparameterization_2_2147972#conv2d_reparameterization_2_2147974#conv2d_reparameterization_2_2147976#conv2d_reparameterization_2_2147978#conv2d_reparameterization_2_2147980#conv2d_reparameterization_2_2147982*
Tin
2	*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:����������� : : *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *a
f\RZ
X__inference_conv2d_reparameterization_2_layer_call_and_return_conditional_losses_2147967�
 max_pooling2d_47/PartitionedCallPartitionedCall<conv2d_reparameterization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������KK * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_2147752�
+depthwise_conv2d_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_47/PartitionedCall:output:0depthwise_conv2d_13_2148003depthwise_conv2d_13_2148005*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������KK�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_depthwise_conv2d_13_layer_call_and_return_conditional_losses_2148002�
 max_pooling2d_48/PartitionedCallPartitionedCall4depthwise_conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_2147764�
"dropout_36/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_48/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_36_layer_call_and_return_conditional_losses_2148021�
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall+dropout_36/StatefulPartitionedCall:output:0conv2d_33_2148036conv2d_33_2148038*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������%%@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_33_layer_call_and_return_conditional_losses_2148035�
 max_pooling2d_49/PartitionedCallPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_2147776�
"dropout_37/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_49/PartitionedCall:output:0#^dropout_36/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_37_layer_call_and_return_conditional_losses_2148054�
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall+dropout_37/StatefulPartitionedCall:output:0conv2d_34_2148069conv2d_34_2148071*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_34_layer_call_and_return_conditional_losses_2148068�
 max_pooling2d_50/PartitionedCallPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_2147788�
"dropout_38/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_50/PartitionedCall:output:0#^dropout_37/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_38_layer_call_and_return_conditional_losses_2148087�
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall+dropout_38/StatefulPartitionedCall:output:0conv2d_35_2148102conv2d_35_2148104*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_35_layer_call_and_return_conditional_losses_2148101�
 max_pooling2d_51/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_2147800�
"dropout_39/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_51/PartitionedCall:output:0#^dropout_38/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_39_layer_call_and_return_conditional_losses_2148120�
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall+dropout_39/StatefulPartitionedCall:output:0conv2d_36_2148135conv2d_36_2148137*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_36_layer_call_and_return_conditional_losses_2148134�
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0conv2d_37_2148153conv2d_37_2148155*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_37_layer_call_and_return_conditional_losses_2148152�
 max_pooling2d_52/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_2147812�
"dropout_40/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_52/PartitionedCall:output:0#^dropout_39/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_40_layer_call_and_return_conditional_losses_2148171�
flatten_7/PartitionedCallPartitionedCall+dropout_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_7_layer_call_and_return_conditional_losses_2148179�
2dense_reparameterization_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0"dense_reparameterization_2_2148326"dense_reparameterization_2_2148328"dense_reparameterization_2_2148330"dense_reparameterization_2_2148332"dense_reparameterization_2_2148334"dense_reparameterization_2_2148336"dense_reparameterization_2_2148338"dense_reparameterization_2_2148340*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������: : *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *`
f[RY
W__inference_dense_reparameterization_2_layer_call_and_return_conditional_losses_2148325�
%one_hot_categorical_7/PartitionedCallPartitionedCall;dense_reparameterization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_one_hot_categorical_7_layer_call_and_return_conditional_losses_2148357}
IdentityIdentity.one_hot_categorical_7/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������|

Identity_1Identity<conv2d_reparameterization_2/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: |

Identity_2Identity<conv2d_reparameterization_2/StatefulPartitionedCall:output:2^NoOp*
T0*
_output_shapes
: {

Identity_3Identity;dense_reparameterization_2/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: {

Identity_4Identity;dense_reparameterization_2/StatefulPartitionedCall:output:2^NoOp*
T0*
_output_shapes
: �
NoOpNoOp"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall4^conv2d_reparameterization_2/StatefulPartitionedCall3^dense_reparameterization_2/StatefulPartitionedCall,^depthwise_conv2d_13/StatefulPartitionedCall#^dropout_36/StatefulPartitionedCall#^dropout_37/StatefulPartitionedCall#^dropout_38/StatefulPartitionedCall#^dropout_39/StatefulPartitionedCall#^dropout_40/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:�����������: : : : : : : : : : : : : : : : : : : : : : : : : :	�: :2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2j
3conv2d_reparameterization_2/StatefulPartitionedCall3conv2d_reparameterization_2/StatefulPartitionedCall2h
2dense_reparameterization_2/StatefulPartitionedCall2dense_reparameterization_2/StatefulPartitionedCall2Z
+depthwise_conv2d_13/StatefulPartitionedCall+depthwise_conv2d_13/StatefulPartitionedCall2H
"dropout_36/StatefulPartitionedCall"dropout_36/StatefulPartitionedCall2H
"dropout_37/StatefulPartitionedCall"dropout_37/StatefulPartitionedCall2H
"dropout_38/StatefulPartitionedCall"dropout_38/StatefulPartitionedCall2H
"dropout_39/StatefulPartitionedCall"dropout_39/StatefulPartitionedCall2H
"dropout_40/StatefulPartitionedCall"dropout_40/StatefulPartitionedCall: 

_output_shapes
::

_output_shapes
: :%!

_output_shapes
:	�:

_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :t p
1
_output_shapes
:�����������
;
_user_specified_name#!conv2d_reparameterization_2_input
�
N
2__inference_max_pooling2d_47_layer_call_fn_2150230

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_2147752�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
e
G__inference_dropout_36_layer_call_and_return_conditional_losses_2148398

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:���������%%�d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:���������%%�"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������%%�:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
�
~
R__inference_one_hot_categorical_7_layer_call_and_return_conditional_losses_2148357

inputs
identity

identity_1|
6tensor_coercible/value/OneHotCategorical/mode/IdentityIdentityinputs*
T0*'
_output_shapes
:����������
>tensor_coercible/value/OneHotCategorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
����������
4tensor_coercible/value/OneHotCategorical/mode/ArgMaxArgMax?tensor_coercible/value/OneHotCategorical/mode/Identity:output:0Gtensor_coercible/value/OneHotCategorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:����������
>tensor_coercible/value/OneHotCategorical/mode/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
?tensor_coercible/value/OneHotCategorical/mode/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    }
;tensor_coercible/value/OneHotCategorical/mode/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :�
5tensor_coercible/value/OneHotCategorical/mode/one_hotOneHot=tensor_coercible/value/OneHotCategorical/mode/ArgMax:output:0Dtensor_coercible/value/OneHotCategorical/mode/one_hot/depth:output:0Gtensor_coercible/value/OneHotCategorical/mode/one_hot/on_value:output:0Htensor_coercible/value/OneHotCategorical/mode/one_hot/off_value:output:0*
T0*'
_output_shapes
:����������
IdentityIdentity>tensor_coercible/value/OneHotCategorical/mode/one_hot:output:0*
T0*'
_output_shapes
:����������

Identity_1Identity>tensor_coercible/value/OneHotCategorical/mode/one_hot:output:0*
T0*'
_output_shapes
:���������"!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�e
�
J__inference_sequential_14_layer_call_and_return_conditional_losses_2148740

inputs=
#conv2d_reparameterization_2_2148653: =
#conv2d_reparameterization_2_2148655: 1
#conv2d_reparameterization_2_2148657: 1
#conv2d_reparameterization_2_2148659: '
#conv2d_reparameterization_2_2148661'
#conv2d_reparameterization_2_2148663'
#conv2d_reparameterization_2_2148665'
#conv2d_reparameterization_2_21486675
depthwise_conv2d_13_2148673: *
depthwise_conv2d_13_2148675:	�,
conv2d_33_2148680:�@
conv2d_33_2148682:@+
conv2d_34_2148687:@@
conv2d_34_2148689:@,
conv2d_35_2148694:@� 
conv2d_35_2148696:	�-
conv2d_36_2148701:�� 
conv2d_36_2148703:	�-
conv2d_37_2148706:�� 
conv2d_37_2148708:	�5
"dense_reparameterization_2_2148714:	�5
"dense_reparameterization_2_2148716:	�0
"dense_reparameterization_2_2148718:0
"dense_reparameterization_2_2148720:&
"dense_reparameterization_2_2148722&
"dense_reparameterization_2_2148724&
"dense_reparameterization_2_2148726&
"dense_reparameterization_2_2148728
identity

identity_1

identity_2

identity_3

identity_4��!conv2d_33/StatefulPartitionedCall�!conv2d_34/StatefulPartitionedCall�!conv2d_35/StatefulPartitionedCall�!conv2d_36/StatefulPartitionedCall�!conv2d_37/StatefulPartitionedCall�3conv2d_reparameterization_2/StatefulPartitionedCall�2dense_reparameterization_2/StatefulPartitionedCall�+depthwise_conv2d_13/StatefulPartitionedCall�
3conv2d_reparameterization_2/StatefulPartitionedCallStatefulPartitionedCallinputs#conv2d_reparameterization_2_2148653#conv2d_reparameterization_2_2148655#conv2d_reparameterization_2_2148657#conv2d_reparameterization_2_2148659#conv2d_reparameterization_2_2148661#conv2d_reparameterization_2_2148663#conv2d_reparameterization_2_2148665#conv2d_reparameterization_2_2148667*
Tin
2	*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:����������� : : *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *a
f\RZ
X__inference_conv2d_reparameterization_2_layer_call_and_return_conditional_losses_2147967�
 max_pooling2d_47/PartitionedCallPartitionedCall<conv2d_reparameterization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������KK * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_2147752�
+depthwise_conv2d_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_47/PartitionedCall:output:0depthwise_conv2d_13_2148673depthwise_conv2d_13_2148675*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������KK�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_depthwise_conv2d_13_layer_call_and_return_conditional_losses_2148002�
 max_pooling2d_48/PartitionedCallPartitionedCall4depthwise_conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_2147764�
dropout_36/PartitionedCallPartitionedCall)max_pooling2d_48/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_36_layer_call_and_return_conditional_losses_2148398�
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall#dropout_36/PartitionedCall:output:0conv2d_33_2148680conv2d_33_2148682*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������%%@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_33_layer_call_and_return_conditional_losses_2148035�
 max_pooling2d_49/PartitionedCallPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_2147776�
dropout_37/PartitionedCallPartitionedCall)max_pooling2d_49/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_37_layer_call_and_return_conditional_losses_2148410�
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall#dropout_37/PartitionedCall:output:0conv2d_34_2148687conv2d_34_2148689*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_34_layer_call_and_return_conditional_losses_2148068�
 max_pooling2d_50/PartitionedCallPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_2147788�
dropout_38/PartitionedCallPartitionedCall)max_pooling2d_50/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_38_layer_call_and_return_conditional_losses_2148422�
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall#dropout_38/PartitionedCall:output:0conv2d_35_2148694conv2d_35_2148696*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_35_layer_call_and_return_conditional_losses_2148101�
 max_pooling2d_51/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_2147800�
dropout_39/PartitionedCallPartitionedCall)max_pooling2d_51/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_39_layer_call_and_return_conditional_losses_2148434�
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall#dropout_39/PartitionedCall:output:0conv2d_36_2148701conv2d_36_2148703*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_36_layer_call_and_return_conditional_losses_2148134�
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0conv2d_37_2148706conv2d_37_2148708*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_37_layer_call_and_return_conditional_losses_2148152�
 max_pooling2d_52/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_2147812�
dropout_40/PartitionedCallPartitionedCall)max_pooling2d_52/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_40_layer_call_and_return_conditional_losses_2148451�
flatten_7/PartitionedCallPartitionedCall#dropout_40/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_7_layer_call_and_return_conditional_losses_2148179�
2dense_reparameterization_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0"dense_reparameterization_2_2148714"dense_reparameterization_2_2148716"dense_reparameterization_2_2148718"dense_reparameterization_2_2148720"dense_reparameterization_2_2148722"dense_reparameterization_2_2148724"dense_reparameterization_2_2148726"dense_reparameterization_2_2148728*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������: : *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *`
f[RY
W__inference_dense_reparameterization_2_layer_call_and_return_conditional_losses_2148325�
%one_hot_categorical_7/PartitionedCallPartitionedCall;dense_reparameterization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_one_hot_categorical_7_layer_call_and_return_conditional_losses_2148484}
IdentityIdentity.one_hot_categorical_7/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������|

Identity_1Identity<conv2d_reparameterization_2/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: |

Identity_2Identity<conv2d_reparameterization_2/StatefulPartitionedCall:output:2^NoOp*
T0*
_output_shapes
: {

Identity_3Identity;dense_reparameterization_2/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: {

Identity_4Identity;dense_reparameterization_2/StatefulPartitionedCall:output:2^NoOp*
T0*
_output_shapes
: �
NoOpNoOp"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall4^conv2d_reparameterization_2/StatefulPartitionedCall3^dense_reparameterization_2/StatefulPartitionedCall,^depthwise_conv2d_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:�����������: : : : : : : : : : : : : : : : : : : : : : : : : :	�: :2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2j
3conv2d_reparameterization_2/StatefulPartitionedCall3conv2d_reparameterization_2/StatefulPartitionedCall2h
2dense_reparameterization_2/StatefulPartitionedCall2dense_reparameterization_2/StatefulPartitionedCall2Z
+depthwise_conv2d_13/StatefulPartitionedCall+depthwise_conv2d_13/StatefulPartitionedCall: 

_output_shapes
::

_output_shapes
: :%!

_output_shapes
:	�:

_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
��
�8
J__inference_sequential_14_layer_call_and_return_conditional_losses_2150057

inputst
Zconv2d_reparameterization_2_independentnormal_sample_normal_sample_readvariableop_resource: }
cconv2d_reparameterization_2_independentnormal_sample_normal_sample_softplus_readvariableop_resource: j
\conv2d_reparameterization_2_independentnormal_sample_1_normal_sample_readvariableop_resource: s
econv2d_reparameterization_2_independentnormal_sample_1_normal_sample_softplus_readvariableop_resource: �
�conv2d_reparameterization_2_kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149788�
�conv2d_reparameterization_2_kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�
�conv2d_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149816�
�conv2d_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xO
5depthwise_conv2d_13_depthwise_readvariableop_resource: B
3depthwise_conv2d_13_biasadd_readvariableop_resource:	�C
(conv2d_33_conv2d_readvariableop_resource:�@7
)conv2d_33_biasadd_readvariableop_resource:@B
(conv2d_34_conv2d_readvariableop_resource:@@7
)conv2d_34_biasadd_readvariableop_resource:@C
(conv2d_35_conv2d_readvariableop_resource:@�8
)conv2d_35_biasadd_readvariableop_resource:	�D
(conv2d_36_conv2d_readvariableop_resource:��8
)conv2d_36_biasadd_readvariableop_resource:	�D
(conv2d_37_conv2d_readvariableop_resource:��8
)conv2d_37_biasadd_readvariableop_resource:	�l
Ydense_reparameterization_2_independentnormal_sample_normal_sample_readvariableop_resource:	�u
bdense_reparameterization_2_independentnormal_sample_normal_sample_softplus_readvariableop_resource:	�i
[dense_reparameterization_2_independentnormal_sample_1_normal_sample_readvariableop_resource:r
ddense_reparameterization_2_independentnormal_sample_1_normal_sample_softplus_readvariableop_resource:�
�dense_reparameterization_2_kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149989�
�dense_reparameterization_2_kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�
�dense_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2150017�
�dense_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity

identity_1

identity_2

identity_3

identity_4�� conv2d_33/BiasAdd/ReadVariableOp�conv2d_33/Conv2D/ReadVariableOp� conv2d_34/BiasAdd/ReadVariableOp�conv2d_34/Conv2D/ReadVariableOp� conv2d_35/BiasAdd/ReadVariableOp�conv2d_35/Conv2D/ReadVariableOp� conv2d_36/BiasAdd/ReadVariableOp�conv2d_36/Conv2D/ReadVariableOp� conv2d_37/BiasAdd/ReadVariableOp�conv2d_37/Conv2D/ReadVariableOp�Qconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/ReadVariableOp�Zconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus/ReadVariableOp�Sconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReadVariableOp�\conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus/ReadVariableOp��conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp��conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp��conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp��conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�Pdense_reparameterization_2/IndependentNormal/sample/Normal/sample/ReadVariableOp�Ydense_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus/ReadVariableOp�Rdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReadVariableOp�[dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus/ReadVariableOp��dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp��dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp��dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp��dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�*depthwise_conv2d_13/BiasAdd/ReadVariableOp�,depthwise_conv2d_13/depthwise/ReadVariableOp�
Aconv2d_reparameterization_2/IndependentNormal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB �
Oconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :�
`conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
Qconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/ReadVariableOpReadVariableOpZconv2d_reparameterization_2_independentnormal_sample_normal_sample_readvariableop_resource*&
_output_shapes
: *
dtype0�
Zconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus/ReadVariableOpReadVariableOpcconv2d_reparameterization_2_independentnormal_sample_normal_sample_softplus_readvariableop_resource*&
_output_shapes
: *
dtype0�
Kconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/SoftplusSoftplusbconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
Hconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
Fconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/addAddV2Qconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/add/x:output:0Yconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus:activations:0*
T0*&
_output_shapes
: �
Rconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"             �
Hconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
Vconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Xconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Xconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Pconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_sliceStridedSlice[conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/shape_as_tensor:output:0_conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice/stack:output:0aconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice/stack_1:output:0aconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Tconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
Jconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : �
Xconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Zconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Zconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Rconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1StridedSlice]conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/shape_as_tensor_1:output:0aconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1/stack:output:0cconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1/stack_1:output:0cconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Sconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB �
Uconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
Pconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgsBroadcastArgs^conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgs/s0_1:output:0Yconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice:output:0*
_output_shapes
:�
Rconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgs_1BroadcastArgsUconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgs:r0:0[conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1:output:0*
_output_shapes
:�
Rconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:�
Nconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Iconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/concatConcatV2[conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/concat/values_0:output:0Wconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgs_1:r0:0Wconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
\conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    �
^conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
lconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalRconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/concat:output:0*
T0**
_output_shapes
: *
dtype0�
[conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/mulMuluconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/RandomStandardNormal:output:0gconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/stddev:output:0*
T0**
_output_shapes
: �
Wconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normalAddV2_conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/mul:z:0econv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/mean:output:0*
T0**
_output_shapes
: �
Fconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/mulMul[conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal:z:0Jconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/add:z:0*
T0**
_output_shapes
: �
Hconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/add_1AddV2Jconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/mul:z:0Yconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/ReadVariableOp:value:0*
T0**
_output_shapes
: �
Pconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"                �
Jconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/ReshapeReshapeLconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/add_1:z:0Yconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/Reshape/shape:output:0*
T0**
_output_shapes
: �
Bconv2d_reparameterization_2/IndependentNormal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
<conv2d_reparameterization_2/IndependentNormal/sample/ReshapeReshapeSconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/Reshape:output:0Kconv2d_reparameterization_2/IndependentNormal/sample/Reshape/shape:output:0*
T0*&
_output_shapes
: �
"conv2d_reparameterization_2/Conv2DConv2DinputsEconv2d_reparameterization_2/IndependentNormal/sample/Reshape:output:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
�
Cconv2d_reparameterization_2/IndependentNormal/sample_1/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB �
Qconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :�
bconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
Sconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReadVariableOpReadVariableOp\conv2d_reparameterization_2_independentnormal_sample_1_normal_sample_readvariableop_resource*
_output_shapes
: *
dtype0�
\conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus/ReadVariableOpReadVariableOpeconv2d_reparameterization_2_independentnormal_sample_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
: *
dtype0�
Mconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/SoftplusSoftplusdconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
: �
Jconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
Hconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/addAddV2Sconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/add/x:output:0[conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus:activations:0*
T0*
_output_shapes
: �
Tconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: �
Jconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
Xconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Zconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Zconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Rconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_sliceStridedSlice]conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/shape_as_tensor:output:0aconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice/stack:output:0cconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice/stack_1:output:0cconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Vconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB: �
Lconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : �
Zconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
\conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
\conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Tconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1StridedSlice_conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/shape_as_tensor_1:output:0cconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1/stack:output:0econv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1/stack_1:output:0econv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Uconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB �
Wconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
Rconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgsBroadcastArgs`conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgs/s0_1:output:0[conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice:output:0*
_output_shapes
:�
Tconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgs_1BroadcastArgsWconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgs:r0:0]conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:�
Tconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:�
Pconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Kconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/concatConcatV2]conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/concat/values_0:output:0Yconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgs_1:r0:0Yconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
^conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    �
`conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
nconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalTconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/concat:output:0*
T0*
_output_shapes

: *
dtype0�
]conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/mulMulwconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0iconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/stddev:output:0*
T0*
_output_shapes

: �
Yconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normalAddV2aconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/mul:z:0gconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/mean:output:0*
T0*
_output_shapes

: �
Hconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/mulMul]conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal:z:0Lconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/add:z:0*
T0*
_output_shapes

: �
Jconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/add_1AddV2Lconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/mul:z:0[conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReadVariableOp:value:0*
T0*
_output_shapes

: �
Rconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       �
Lconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReshapeReshapeNconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/add_1:z:0[conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

: �
Dconv2d_reparameterization_2/IndependentNormal/sample_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB: �
>conv2d_reparameterization_2/IndependentNormal/sample_1/ReshapeReshapeUconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Reshape:output:0Mconv2d_reparameterization_2/IndependentNormal/sample_1/Reshape/shape:output:0*
T0*
_output_shapes
: �
#conv2d_reparameterization_2/BiasAddBiasAdd+conv2d_reparameterization_2/Conv2D:output:0Gconv2d_reparameterization_2/IndependentNormal/sample_1/Reshape:output:0*
T0*1
_output_shapes
:����������� �
 conv2d_reparameterization_2/ReluRelu,conv2d_reparameterization_2/BiasAdd:output:0*
T0*1
_output_shapes
:����������� �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOpcconv2d_reparameterization_2_independentnormal_sample_normal_sample_softplus_readvariableop_resource*&
_output_shapes
: *
dtype0�
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log�conv2d_reparameterization_2_kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149788*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOpZconv2d_reparameterization_2_independentnormal_sample_normal_sample_readvariableop_resource*&
_output_shapes
: *
dtype0�
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0�conv2d_reparameterization_2_kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149788*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv�conv2d_reparameterization_2_kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�conv2d_reparameterization_2_kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149788*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"�����������������
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: j
%conv2d_reparameterization_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * H[F�
#conv2d_reparameterization_2/truedivRealDiv�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0.conv2d_reparameterization_2/truediv/y:output:0*
T0*
_output_shapes
: �
-conv2d_reparameterization_2/divergence_kernelIdentity'conv2d_reparameterization_2/truediv:z:0*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOpeconv2d_reparameterization_2_independentnormal_sample_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
: *
dtype0�
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log�conv2d_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149816*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp\conv2d_reparameterization_2_independentnormal_sample_1_normal_sample_readvariableop_resource*
_output_shapes
: *
dtype0�
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0�conv2d_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149816*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv�conv2d_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�conv2d_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149816*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: l
'conv2d_reparameterization_2/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 * H[F�
%conv2d_reparameterization_2/truediv_1RealDiv�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:00conv2d_reparameterization_2/truediv_1/y:output:0*
T0*
_output_shapes
: �
+conv2d_reparameterization_2/divergence_biasIdentity)conv2d_reparameterization_2/truediv_1:z:0*
T0*
_output_shapes
: �
max_pooling2d_47/MaxPoolMaxPool.conv2d_reparameterization_2/Relu:activations:0*/
_output_shapes
:���������KK *
ksize
*
paddingVALID*
strides
�
,depthwise_conv2d_13/depthwise/ReadVariableOpReadVariableOp5depthwise_conv2d_13_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype0|
#depthwise_conv2d_13/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             |
+depthwise_conv2d_13/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
depthwise_conv2d_13/depthwiseDepthwiseConv2dNative!max_pooling2d_47/MaxPool:output:04depthwise_conv2d_13/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
�
*depthwise_conv2d_13/BiasAdd/ReadVariableOpReadVariableOp3depthwise_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
depthwise_conv2d_13/BiasAddBiasAdd&depthwise_conv2d_13/depthwise:output:02depthwise_conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK��
)depthwise_conv2d_13/activation_16/SigmoidSigmoid$depthwise_conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK��
%depthwise_conv2d_13/activation_16/mulMul-depthwise_conv2d_13/activation_16/Sigmoid:y:0$depthwise_conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK��
max_pooling2d_48/MaxPoolMaxPool)depthwise_conv2d_13/activation_16/mul:z:0*0
_output_shapes
:���������%%�*
ksize
*
paddingVALID*
strides
}
dropout_36/IdentityIdentity!max_pooling2d_48/MaxPool:output:0*
T0*0
_output_shapes
:���������%%��
conv2d_33/Conv2D/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
conv2d_33/Conv2DConv2Ddropout_36/Identity:output:0'conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������%%@*
paddingSAME*
strides
�
 conv2d_33/BiasAdd/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_33/BiasAddBiasAddconv2d_33/Conv2D:output:0(conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������%%@�
conv2d_33/activation_16/SigmoidSigmoidconv2d_33/BiasAdd:output:0*
T0*/
_output_shapes
:���������%%@�
conv2d_33/activation_16/mulMul#conv2d_33/activation_16/Sigmoid:y:0conv2d_33/BiasAdd:output:0*
T0*/
_output_shapes
:���������%%@�
max_pooling2d_49/MaxPoolMaxPoolconv2d_33/activation_16/mul:z:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
|
dropout_37/IdentityIdentity!max_pooling2d_49/MaxPool:output:0*
T0*/
_output_shapes
:���������@�
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_34/Conv2DConv2Ddropout_37/Identity:output:0'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
conv2d_34/activation_16/SigmoidSigmoidconv2d_34/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
conv2d_34/activation_16/mulMul#conv2d_34/activation_16/Sigmoid:y:0conv2d_34/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
max_pooling2d_50/MaxPoolMaxPoolconv2d_34/activation_16/mul:z:0*/
_output_shapes
:���������		@*
ksize
*
paddingVALID*
strides
|
dropout_38/IdentityIdentity!max_pooling2d_50/MaxPool:output:0*
T0*/
_output_shapes
:���������		@�
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_35/Conv2DConv2Ddropout_38/Identity:output:0'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
�
 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		��
conv2d_35/activation_16/SigmoidSigmoidconv2d_35/BiasAdd:output:0*
T0*0
_output_shapes
:���������		��
conv2d_35/activation_16/mulMul#conv2d_35/activation_16/Sigmoid:y:0conv2d_35/BiasAdd:output:0*
T0*0
_output_shapes
:���������		��
max_pooling2d_51/MaxPoolMaxPoolconv2d_35/activation_16/mul:z:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
}
dropout_39/IdentityIdentity!max_pooling2d_51/MaxPool:output:0*
T0*0
_output_shapes
:�����������
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_36/Conv2DConv2Ddropout_39/Identity:output:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
conv2d_36/activation_16/SigmoidSigmoidconv2d_36/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
conv2d_36/activation_16/mulMul#conv2d_36/activation_16/Sigmoid:y:0conv2d_36/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_37/Conv2DConv2Dconv2d_36/activation_16/mul:z:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
conv2d_37/activation_16/SigmoidSigmoidconv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
conv2d_37/activation_16/mulMul#conv2d_37/activation_16/Sigmoid:y:0conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
max_pooling2d_52/MaxPoolMaxPoolconv2d_37/activation_16/mul:z:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
}
dropout_40/IdentityIdentity!max_pooling2d_52/MaxPool:output:0*
T0*0
_output_shapes
:����������`
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten_7/ReshapeReshapedropout_40/Identity:output:0flatten_7/Const:output:0*
T0*(
_output_shapes
:�����������
@dense_reparameterization_2/IndependentNormal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB �
Ndense_reparameterization_2/IndependentNormal/sample/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :�
_dense_reparameterization_2/IndependentNormal/sample/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
Pdense_reparameterization_2/IndependentNormal/sample/Normal/sample/ReadVariableOpReadVariableOpYdense_reparameterization_2_independentnormal_sample_normal_sample_readvariableop_resource*
_output_shapes
:	�*
dtype0�
Ydense_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus/ReadVariableOpReadVariableOpbdense_reparameterization_2_independentnormal_sample_normal_sample_softplus_readvariableop_resource*
_output_shapes
:	�*
dtype0�
Jdense_reparameterization_2/IndependentNormal/sample/Normal/sample/SoftplusSoftplusadense_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
Gdense_reparameterization_2/IndependentNormal/sample/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
Edense_reparameterization_2/IndependentNormal/sample/Normal/sample/addAddV2Pdense_reparameterization_2/IndependentNormal/sample/Normal/sample/add/x:output:0Xdense_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus:activations:0*
T0*
_output_shapes
:	��
Qdense_reparameterization_2/IndependentNormal/sample/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"      �
Gdense_reparameterization_2/IndependentNormal/sample/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
Udense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Wdense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Wdense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Odense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_sliceStridedSliceZdense_reparameterization_2/IndependentNormal/sample/Normal/sample/shape_as_tensor:output:0^dense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice/stack:output:0`dense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice/stack_1:output:0`dense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Sdense_reparameterization_2/IndependentNormal/sample/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"      �
Idense_reparameterization_2/IndependentNormal/sample/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : �
Wdense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Ydense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Ydense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Qdense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1StridedSlice\dense_reparameterization_2/IndependentNormal/sample/Normal/sample/shape_as_tensor_1:output:0`dense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1/stack:output:0bdense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1/stack_1:output:0bdense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Rdense_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB �
Tdense_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
Odense_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgsBroadcastArgs]dense_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgs/s0_1:output:0Xdense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice:output:0*
_output_shapes
:�
Qdense_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgs_1BroadcastArgsTdense_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgs:r0:0Zdense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1:output:0*
_output_shapes
:�
Qdense_reparameterization_2/IndependentNormal/sample/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:�
Mdense_reparameterization_2/IndependentNormal/sample/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Hdense_reparameterization_2/IndependentNormal/sample/Normal/sample/concatConcatV2Zdense_reparameterization_2/IndependentNormal/sample/Normal/sample/concat/values_0:output:0Vdense_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgs_1:r0:0Vdense_reparameterization_2/IndependentNormal/sample/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
[dense_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    �
]dense_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
kdense_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalQdense_reparameterization_2/IndependentNormal/sample/Normal/sample/concat:output:0*
T0*#
_output_shapes
:�*
dtype0�
Zdense_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/mulMultdense_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/RandomStandardNormal:output:0fdense_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/stddev:output:0*
T0*#
_output_shapes
:��
Vdense_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normalAddV2^dense_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/mul:z:0ddense_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/mean:output:0*
T0*#
_output_shapes
:��
Edense_reparameterization_2/IndependentNormal/sample/Normal/sample/mulMulZdense_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal:z:0Idense_reparameterization_2/IndependentNormal/sample/Normal/sample/add:z:0*
T0*#
_output_shapes
:��
Gdense_reparameterization_2/IndependentNormal/sample/Normal/sample/add_1AddV2Idense_reparameterization_2/IndependentNormal/sample/Normal/sample/mul:z:0Xdense_reparameterization_2/IndependentNormal/sample/Normal/sample/ReadVariableOp:value:0*
T0*#
_output_shapes
:��
Odense_reparameterization_2/IndependentNormal/sample/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
Idense_reparameterization_2/IndependentNormal/sample/Normal/sample/ReshapeReshapeKdense_reparameterization_2/IndependentNormal/sample/Normal/sample/add_1:z:0Xdense_reparameterization_2/IndependentNormal/sample/Normal/sample/Reshape/shape:output:0*
T0*#
_output_shapes
:��
Adense_reparameterization_2/IndependentNormal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
;dense_reparameterization_2/IndependentNormal/sample/ReshapeReshapeRdense_reparameterization_2/IndependentNormal/sample/Normal/sample/Reshape:output:0Jdense_reparameterization_2/IndependentNormal/sample/Reshape/shape:output:0*
T0*
_output_shapes
:	��
!dense_reparameterization_2/MatMulMatMulflatten_7/Reshape:output:0Ddense_reparameterization_2/IndependentNormal/sample/Reshape:output:0*
T0*'
_output_shapes
:����������
Bdense_reparameterization_2/IndependentNormal/sample_1/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB �
Pdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :�
adense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
Rdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReadVariableOpReadVariableOp[dense_reparameterization_2_independentnormal_sample_1_normal_sample_readvariableop_resource*
_output_shapes
:*
dtype0�
[dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus/ReadVariableOpReadVariableOpddense_reparameterization_2_independentnormal_sample_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
:*
dtype0�
Ldense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/SoftplusSoftpluscdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:�
Idense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
Gdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/addAddV2Rdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/add/x:output:0Zdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus:activations:0*
T0*
_output_shapes
:�
Sdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:�
Idense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
Wdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Ydense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Ydense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Qdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_sliceStridedSlice\dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/shape_as_tensor:output:0`dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice/stack:output:0bdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice/stack_1:output:0bdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Udense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:�
Kdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : �
Ydense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
[dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
[dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Sdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1StridedSlice^dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/shape_as_tensor_1:output:0bdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1/stack:output:0ddense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1/stack_1:output:0ddense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Tdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB �
Vdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
Qdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgsBroadcastArgs_dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgs/s0_1:output:0Zdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice:output:0*
_output_shapes
:�
Sdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgs_1BroadcastArgsVdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgs:r0:0\dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:�
Sdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:�
Odense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Jdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/concatConcatV2\dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/concat/values_0:output:0Xdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgs_1:r0:0Xdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
]dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    �
_dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
mdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalSdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/concat:output:0*
T0*
_output_shapes

:*
dtype0�
\dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/mulMulvdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0hdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/stddev:output:0*
T0*
_output_shapes

:�
Xdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normalAddV2`dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/mul:z:0fdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/mean:output:0*
T0*
_output_shapes

:�
Gdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/mulMul\dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal:z:0Kdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/add:z:0*
T0*
_output_shapes

:�
Idense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/add_1AddV2Kdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/mul:z:0Zdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReadVariableOp:value:0*
T0*
_output_shapes

:�
Qdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
Kdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReshapeReshapeMdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/add_1:z:0Zdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

:�
Cdense_reparameterization_2/IndependentNormal/sample_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
=dense_reparameterization_2/IndependentNormal/sample_1/ReshapeReshapeTdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Reshape:output:0Ldense_reparameterization_2/IndependentNormal/sample_1/Reshape/shape:output:0*
T0*
_output_shapes
:�
"dense_reparameterization_2/BiasAddBiasAdd+dense_reparameterization_2/MatMul:product:0Fdense_reparameterization_2/IndependentNormal/sample_1/Reshape:output:0*
T0*'
_output_shapes
:����������
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOpbdense_reparameterization_2_independentnormal_sample_normal_sample_softplus_readvariableop_resource*
_output_shapes
:	�*
dtype0�
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log�dense_reparameterization_2_kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149989*
T0*
_output_shapes
: �
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOpYdense_reparameterization_2_independentnormal_sample_normal_sample_readvariableop_resource*
_output_shapes
:	�*
dtype0�
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0�dense_reparameterization_2_kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149989*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv�dense_reparameterization_2_kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�dense_reparameterization_2_kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149989*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"���������
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: i
$dense_reparameterization_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * H[F�
"dense_reparameterization_2/truedivRealDiv�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0-dense_reparameterization_2/truediv/y:output:0*
T0*
_output_shapes
: �
,dense_reparameterization_2/divergence_kernelIdentity&dense_reparameterization_2/truediv:z:0*
T0*
_output_shapes
: �
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOpddense_reparameterization_2_independentnormal_sample_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
:*
dtype0�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log�dense_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2150017*
T0*
_output_shapes
: �
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp[dense_reparameterization_2_independentnormal_sample_1_normal_sample_readvariableop_resource*
_output_shapes
:*
dtype0�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0�dense_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2150017*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv�dense_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�dense_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2150017*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: k
&dense_reparameterization_2/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 * H[F�
$dense_reparameterization_2/truediv_1RealDiv�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0/dense_reparameterization_2/truediv_1/y:output:0*
T0*
_output_shapes
: �
*dense_reparameterization_2/divergence_biasIdentity(dense_reparameterization_2/truediv_1:z:0*
T0*
_output_shapes
: �
Lone_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/IdentityIdentity+dense_reparameterization_2/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Tone_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
����������
Jone_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/ArgMaxArgMaxUone_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/Identity:output:0]one_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:����������
Tone_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Uone_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Qone_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :�
Kone_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/one_hotOneHotSone_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/ArgMax:output:0Zone_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/one_hot/depth:output:0]one_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/one_hot/on_value:output:0^one_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/one_hot/off_value:output:0*
T0*'
_output_shapes
:����������
IdentityIdentityTone_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/one_hot:output:0^NoOp*
T0*'
_output_shapes
:���������v

Identity_1Identity6conv2d_reparameterization_2/divergence_kernel:output:0^NoOp*
T0*
_output_shapes
: t

Identity_2Identity4conv2d_reparameterization_2/divergence_bias:output:0^NoOp*
T0*
_output_shapes
: u

Identity_3Identity5dense_reparameterization_2/divergence_kernel:output:0^NoOp*
T0*
_output_shapes
: s

Identity_4Identity3dense_reparameterization_2/divergence_bias:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp!^conv2d_33/BiasAdd/ReadVariableOp ^conv2d_33/Conv2D/ReadVariableOp!^conv2d_34/BiasAdd/ReadVariableOp ^conv2d_34/Conv2D/ReadVariableOp!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOpR^conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/ReadVariableOp[^conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus/ReadVariableOpT^conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReadVariableOp]^conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus/ReadVariableOp�^conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�^conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�^conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�^conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpQ^dense_reparameterization_2/IndependentNormal/sample/Normal/sample/ReadVariableOpZ^dense_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus/ReadVariableOpS^dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReadVariableOp\^dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus/ReadVariableOp�^dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�^dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�^dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�^dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp+^depthwise_conv2d_13/BiasAdd/ReadVariableOp-^depthwise_conv2d_13/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:�����������: : : : : : : : : : : : : : : : : : : : : : : : : :	�: :2D
 conv2d_33/BiasAdd/ReadVariableOp conv2d_33/BiasAdd/ReadVariableOp2B
conv2d_33/Conv2D/ReadVariableOpconv2d_33/Conv2D/ReadVariableOp2D
 conv2d_34/BiasAdd/ReadVariableOp conv2d_34/BiasAdd/ReadVariableOp2B
conv2d_34/Conv2D/ReadVariableOpconv2d_34/Conv2D/ReadVariableOp2D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2D
 conv2d_36/BiasAdd/ReadVariableOp conv2d_36/BiasAdd/ReadVariableOp2B
conv2d_36/Conv2D/ReadVariableOpconv2d_36/Conv2D/ReadVariableOp2D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp2�
Qconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/ReadVariableOpQconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/ReadVariableOp2�
Zconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus/ReadVariableOpZconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus/ReadVariableOp2�
Sconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReadVariableOpSconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReadVariableOp2�
\conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus/ReadVariableOp\conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus/ReadVariableOp2�
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2�
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2�
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2�
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2�
Pdense_reparameterization_2/IndependentNormal/sample/Normal/sample/ReadVariableOpPdense_reparameterization_2/IndependentNormal/sample/Normal/sample/ReadVariableOp2�
Ydense_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus/ReadVariableOpYdense_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus/ReadVariableOp2�
Rdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReadVariableOpRdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReadVariableOp2�
[dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus/ReadVariableOp[dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus/ReadVariableOp2�
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2�
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2X
*depthwise_conv2d_13/BiasAdd/ReadVariableOp*depthwise_conv2d_13/BiasAdd/ReadVariableOp2\
,depthwise_conv2d_13/depthwise/ReadVariableOp,depthwise_conv2d_13/depthwise/ReadVariableOp: 

_output_shapes
::

_output_shapes
: :%!

_output_shapes
:	�:

_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

f
G__inference_dropout_38_layer_call_and_return_conditional_losses_2148087

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������		@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������		@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������		@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:���������		@i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:���������		@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������		@:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
e
G__inference_dropout_40_layer_call_and_return_conditional_losses_2148451

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:����������d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
/__inference_sequential_14_layer_call_fn_2149312

inputs!
unknown: #
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3
	unknown_4
	unknown_5
	unknown_6#
	unknown_7: 
	unknown_8:	�$
	unknown_9:�@

unknown_10:@$

unknown_11:@@

unknown_12:@%

unknown_13:@�

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�

unknown_19:	�

unknown_20:	�

unknown_21:

unknown_22:

unknown_23

unknown_24

unknown_25

unknown_26
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout	
2*
_collective_manager_ids
 */
_output_shapes
:���������: : : : *6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_14_layer_call_and_return_conditional_losses_2148740o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:�����������: : : : : : : : : : : : : : : : : : : : : : : : : :	�: :22
StatefulPartitionedCallStatefulPartitionedCall: 

_output_shapes
::

_output_shapes
: :%!

_output_shapes
:	�:

_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
e
,__inference_dropout_36_layer_call_fn_2150273

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_36_layer_call_and_return_conditional_losses_2148021x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������%%�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������%%�22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
�
G
+__inference_flatten_7_layer_call_fn_2150553

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_7_layer_call_and_return_conditional_losses_2148179a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_35_layer_call_and_return_conditional_losses_2148101

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�m
activation_16/SigmoidSigmoidBiasAdd:output:0*
T0*0
_output_shapes
:���������		��
activation_16/mulMulactivation_16/Sigmoid:y:0BiasAdd:output:0*
T0*0
_output_shapes
:���������		�m
IdentityIdentityactivation_16/mul:z:0^NoOp*
T0*0
_output_shapes
:���������		�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������		@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
n
R__inference_one_hot_categorical_7_layer_call_and_return_conditional_losses_2150751

inputs
identity|
6tensor_coercible/value/OneHotCategorical/mode/IdentityIdentityinputs*
T0*'
_output_shapes
:����������
>tensor_coercible/value/OneHotCategorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
����������
4tensor_coercible/value/OneHotCategorical/mode/ArgMaxArgMax?tensor_coercible/value/OneHotCategorical/mode/Identity:output:0Gtensor_coercible/value/OneHotCategorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:����������
>tensor_coercible/value/OneHotCategorical/mode/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
?tensor_coercible/value/OneHotCategorical/mode/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    }
;tensor_coercible/value/OneHotCategorical/mode/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :�
5tensor_coercible/value/OneHotCategorical/mode/one_hotOneHot=tensor_coercible/value/OneHotCategorical/mode/ArgMax:output:0Dtensor_coercible/value/OneHotCategorical/mode/one_hot/depth:output:0Gtensor_coercible/value/OneHotCategorical/mode/one_hot/on_value:output:0Htensor_coercible/value/OneHotCategorical/mode/one_hot/off_value:output:0*
T0*'
_output_shapes
:����������
IdentityIdentity>tensor_coercible/value/OneHotCategorical/mode/one_hot:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

f
G__inference_dropout_37_layer_call_and_return_conditional_losses_2150348

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:���������@i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
/__inference_sequential_14_layer_call_fn_2148803%
!conv2d_reparameterization_2_input!
unknown: #
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3
	unknown_4
	unknown_5
	unknown_6#
	unknown_7: 
	unknown_8:	�$
	unknown_9:�@

unknown_10:@$

unknown_11:@@

unknown_12:@%

unknown_13:@�

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�

unknown_19:	�

unknown_20:	�

unknown_21:

unknown_22:

unknown_23

unknown_24

unknown_25

unknown_26
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall!conv2d_reparameterization_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout	
2*
_collective_manager_ids
 */
_output_shapes
:���������: : : : *6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_14_layer_call_and_return_conditional_losses_2148740o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:�����������: : : : : : : : : : : : : : : : : : : : : : : : : :	�: :22
StatefulPartitionedCallStatefulPartitionedCall: 

_output_shapes
::

_output_shapes
: :%!

_output_shapes
:	�:

_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :t p
1
_output_shapes
:�����������
;
_user_specified_name#!conv2d_reparameterization_2_input
�
i
M__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_2147788

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
N
2__inference_max_pooling2d_48_layer_call_fn_2150263

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_2147764�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�m
�
J__inference_sequential_14_layer_call_and_return_conditional_losses_2148585

inputs=
#conv2d_reparameterization_2_2148498: =
#conv2d_reparameterization_2_2148500: 1
#conv2d_reparameterization_2_2148502: 1
#conv2d_reparameterization_2_2148504: '
#conv2d_reparameterization_2_2148506'
#conv2d_reparameterization_2_2148508'
#conv2d_reparameterization_2_2148510'
#conv2d_reparameterization_2_21485125
depthwise_conv2d_13_2148518: *
depthwise_conv2d_13_2148520:	�,
conv2d_33_2148525:�@
conv2d_33_2148527:@+
conv2d_34_2148532:@@
conv2d_34_2148534:@,
conv2d_35_2148539:@� 
conv2d_35_2148541:	�-
conv2d_36_2148546:�� 
conv2d_36_2148548:	�-
conv2d_37_2148551:�� 
conv2d_37_2148553:	�5
"dense_reparameterization_2_2148559:	�5
"dense_reparameterization_2_2148561:	�0
"dense_reparameterization_2_2148563:0
"dense_reparameterization_2_2148565:&
"dense_reparameterization_2_2148567&
"dense_reparameterization_2_2148569&
"dense_reparameterization_2_2148571&
"dense_reparameterization_2_2148573
identity

identity_1

identity_2

identity_3

identity_4��!conv2d_33/StatefulPartitionedCall�!conv2d_34/StatefulPartitionedCall�!conv2d_35/StatefulPartitionedCall�!conv2d_36/StatefulPartitionedCall�!conv2d_37/StatefulPartitionedCall�3conv2d_reparameterization_2/StatefulPartitionedCall�2dense_reparameterization_2/StatefulPartitionedCall�+depthwise_conv2d_13/StatefulPartitionedCall�"dropout_36/StatefulPartitionedCall�"dropout_37/StatefulPartitionedCall�"dropout_38/StatefulPartitionedCall�"dropout_39/StatefulPartitionedCall�"dropout_40/StatefulPartitionedCall�
3conv2d_reparameterization_2/StatefulPartitionedCallStatefulPartitionedCallinputs#conv2d_reparameterization_2_2148498#conv2d_reparameterization_2_2148500#conv2d_reparameterization_2_2148502#conv2d_reparameterization_2_2148504#conv2d_reparameterization_2_2148506#conv2d_reparameterization_2_2148508#conv2d_reparameterization_2_2148510#conv2d_reparameterization_2_2148512*
Tin
2	*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:����������� : : *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *a
f\RZ
X__inference_conv2d_reparameterization_2_layer_call_and_return_conditional_losses_2147967�
 max_pooling2d_47/PartitionedCallPartitionedCall<conv2d_reparameterization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������KK * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_2147752�
+depthwise_conv2d_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_47/PartitionedCall:output:0depthwise_conv2d_13_2148518depthwise_conv2d_13_2148520*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������KK�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_depthwise_conv2d_13_layer_call_and_return_conditional_losses_2148002�
 max_pooling2d_48/PartitionedCallPartitionedCall4depthwise_conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_2147764�
"dropout_36/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_48/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_36_layer_call_and_return_conditional_losses_2148021�
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall+dropout_36/StatefulPartitionedCall:output:0conv2d_33_2148525conv2d_33_2148527*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������%%@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_33_layer_call_and_return_conditional_losses_2148035�
 max_pooling2d_49/PartitionedCallPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_2147776�
"dropout_37/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_49/PartitionedCall:output:0#^dropout_36/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_37_layer_call_and_return_conditional_losses_2148054�
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall+dropout_37/StatefulPartitionedCall:output:0conv2d_34_2148532conv2d_34_2148534*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_34_layer_call_and_return_conditional_losses_2148068�
 max_pooling2d_50/PartitionedCallPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_2147788�
"dropout_38/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_50/PartitionedCall:output:0#^dropout_37/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_38_layer_call_and_return_conditional_losses_2148087�
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall+dropout_38/StatefulPartitionedCall:output:0conv2d_35_2148539conv2d_35_2148541*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_35_layer_call_and_return_conditional_losses_2148101�
 max_pooling2d_51/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_2147800�
"dropout_39/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_51/PartitionedCall:output:0#^dropout_38/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_39_layer_call_and_return_conditional_losses_2148120�
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall+dropout_39/StatefulPartitionedCall:output:0conv2d_36_2148546conv2d_36_2148548*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_36_layer_call_and_return_conditional_losses_2148134�
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0conv2d_37_2148551conv2d_37_2148553*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_37_layer_call_and_return_conditional_losses_2148152�
 max_pooling2d_52/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_2147812�
"dropout_40/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_52/PartitionedCall:output:0#^dropout_39/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_40_layer_call_and_return_conditional_losses_2148171�
flatten_7/PartitionedCallPartitionedCall+dropout_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_7_layer_call_and_return_conditional_losses_2148179�
2dense_reparameterization_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0"dense_reparameterization_2_2148559"dense_reparameterization_2_2148561"dense_reparameterization_2_2148563"dense_reparameterization_2_2148565"dense_reparameterization_2_2148567"dense_reparameterization_2_2148569"dense_reparameterization_2_2148571"dense_reparameterization_2_2148573*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������: : *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *`
f[RY
W__inference_dense_reparameterization_2_layer_call_and_return_conditional_losses_2148325�
%one_hot_categorical_7/PartitionedCallPartitionedCall;dense_reparameterization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_one_hot_categorical_7_layer_call_and_return_conditional_losses_2148357}
IdentityIdentity.one_hot_categorical_7/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������|

Identity_1Identity<conv2d_reparameterization_2/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: |

Identity_2Identity<conv2d_reparameterization_2/StatefulPartitionedCall:output:2^NoOp*
T0*
_output_shapes
: {

Identity_3Identity;dense_reparameterization_2/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: {

Identity_4Identity;dense_reparameterization_2/StatefulPartitionedCall:output:2^NoOp*
T0*
_output_shapes
: �
NoOpNoOp"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall4^conv2d_reparameterization_2/StatefulPartitionedCall3^dense_reparameterization_2/StatefulPartitionedCall,^depthwise_conv2d_13/StatefulPartitionedCall#^dropout_36/StatefulPartitionedCall#^dropout_37/StatefulPartitionedCall#^dropout_38/StatefulPartitionedCall#^dropout_39/StatefulPartitionedCall#^dropout_40/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:�����������: : : : : : : : : : : : : : : : : : : : : : : : : :	�: :2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2j
3conv2d_reparameterization_2/StatefulPartitionedCall3conv2d_reparameterization_2/StatefulPartitionedCall2h
2dense_reparameterization_2/StatefulPartitionedCall2dense_reparameterization_2/StatefulPartitionedCall2Z
+depthwise_conv2d_13/StatefulPartitionedCall+depthwise_conv2d_13/StatefulPartitionedCall2H
"dropout_36/StatefulPartitionedCall"dropout_36/StatefulPartitionedCall2H
"dropout_37/StatefulPartitionedCall"dropout_37/StatefulPartitionedCall2H
"dropout_38/StatefulPartitionedCall"dropout_38/StatefulPartitionedCall2H
"dropout_39/StatefulPartitionedCall"dropout_39/StatefulPartitionedCall2H
"dropout_40/StatefulPartitionedCall"dropout_40/StatefulPartitionedCall: 

_output_shapes
::

_output_shapes
: :%!

_output_shapes
:	�:

_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
b
F__inference_flatten_7_layer_call_and_return_conditional_losses_2150559

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

f
G__inference_dropout_37_layer_call_and_return_conditional_losses_2148054

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:���������@i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
+__inference_conv2d_36_layer_call_fn_2150478

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_36_layer_call_and_return_conditional_losses_2148134x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
X__inference_conv2d_reparameterization_2_layer_call_and_return_conditional_losses_2147967

inputs�
iindependentnormal_constructed_at_conv2d_reparameterization_2_sample_normal_sample_readvariableop_resource: �
rindependentnormal_constructed_at_conv2d_reparameterization_2_sample_normal_sample_softplus_readvariableop_resource: y
kindependentnormal_constructed_at_conv2d_reparameterization_2_sample_1_normal_sample_readvariableop_resource: �
tindependentnormal_constructed_at_conv2d_reparameterization_2_sample_1_normal_sample_softplus_readvariableop_resource: �
�kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2147908�
�kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�
�kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2147936�
�kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity

identity_1

identity_2��`IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/ReadVariableOp�iIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOp�bIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/ReadVariableOp�kIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp��KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp��KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp��KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp��KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�
PIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB �
^IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :�
oIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
`IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/ReadVariableOpReadVariableOpiindependentnormal_constructed_at_conv2d_reparameterization_2_sample_normal_sample_readvariableop_resource*&
_output_shapes
: *
dtype0�
iIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOpReadVariableOprindependentnormal_constructed_at_conv2d_reparameterization_2_sample_normal_sample_softplus_readvariableop_resource*&
_output_shapes
: *
dtype0�
ZIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/SoftplusSoftplusqIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
WIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
UIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/addAddV2`IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/add/x:output:0hIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Softplus:activations:0*
T0*&
_output_shapes
: �
aIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"             �
WIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
eIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
gIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
gIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
_IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_sliceStridedSlicejIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/shape_as_tensor:output:0nIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice/stack:output:0pIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice/stack_1:output:0pIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
cIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
YIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : �
gIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
iIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
iIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
aIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice_1StridedSlicelIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/shape_as_tensor_1:output:0pIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice_1/stack:output:0rIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice_1/stack_1:output:0rIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
bIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB �
dIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
_IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/BroadcastArgsBroadcastArgsmIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/BroadcastArgs/s0_1:output:0hIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice:output:0*
_output_shapes
:�
aIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/BroadcastArgs_1BroadcastArgsdIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/BroadcastArgs:r0:0jIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice_1:output:0*
_output_shapes
:�
aIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:�
]IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
XIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/concatConcatV2jIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/concat/values_0:output:0fIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/BroadcastArgs_1:r0:0fIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
kIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    �
mIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
{IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalaIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/concat:output:0*
T0**
_output_shapes
: *
dtype0�
jIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal/mulMul�IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal/RandomStandardNormal:output:0vIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal/stddev:output:0*
T0**
_output_shapes
: �
fIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normalAddV2nIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal/mul:z:0tIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal/mean:output:0*
T0**
_output_shapes
: �
UIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/mulMuljIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal:z:0YIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/add:z:0*
T0**
_output_shapes
: �
WIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/add_1AddV2YIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/mul:z:0hIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/ReadVariableOp:value:0*
T0**
_output_shapes
: �
_IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"                �
YIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/ReshapeReshape[IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/add_1:z:0hIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Reshape/shape:output:0*
T0**
_output_shapes
: �
QIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
KIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/ReshapeReshapebIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Reshape:output:0ZIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Reshape/shape:output:0*
T0*&
_output_shapes
: �
Conv2DConv2DinputsTIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Reshape:output:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
�
RIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB �
`IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :�
qIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
bIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/ReadVariableOpReadVariableOpkindependentnormal_constructed_at_conv2d_reparameterization_2_sample_1_normal_sample_readvariableop_resource*
_output_shapes
: *
dtype0�
kIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOpReadVariableOptindependentnormal_constructed_at_conv2d_reparameterization_2_sample_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
: *
dtype0�
\IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/SoftplusSoftplussIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
: �
YIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
WIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/addAddV2bIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/add/x:output:0jIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Softplus:activations:0*
T0*
_output_shapes
: �
cIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: �
YIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
gIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
iIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
iIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
aIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_sliceStridedSlicelIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/shape_as_tensor:output:0pIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice/stack:output:0rIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice/stack_1:output:0rIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
eIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB: �
[IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : �
iIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
kIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
kIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
cIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice_1StridedSlicenIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/shape_as_tensor_1:output:0rIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack:output:0tIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack_1:output:0tIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
dIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB �
fIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
aIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/BroadcastArgsBroadcastArgsoIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/BroadcastArgs/s0_1:output:0jIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice:output:0*
_output_shapes
:�
cIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/BroadcastArgs_1BroadcastArgsfIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/BroadcastArgs:r0:0lIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:�
cIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:�
_IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
ZIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/concatConcatV2lIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/concat/values_0:output:0hIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/BroadcastArgs_1:r0:0hIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
mIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    �
oIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
}IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalcIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/concat:output:0*
T0*
_output_shapes

: *
dtype0�
lIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal/mulMul�IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0xIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal/stddev:output:0*
T0*
_output_shapes

: �
hIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normalAddV2pIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal/mul:z:0vIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal/mean:output:0*
T0*
_output_shapes

: �
WIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/mulMullIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal:z:0[IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/add:z:0*
T0*
_output_shapes

: �
YIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/add_1AddV2[IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/mul:z:0jIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/ReadVariableOp:value:0*
T0*
_output_shapes

: �
aIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       �
[IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/ReshapeReshape]IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/add_1:z:0jIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

: �
SIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB: �
MIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/ReshapeReshapedIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Reshape:output:0\IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Reshape/shape:output:0*
T0*
_output_shapes
: �
BiasAddBiasAddConv2D:output:0VIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Reshape:output:0*
T0*1
_output_shapes
:����������� Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:����������� �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOprindependentnormal_constructed_at_conv2d_reparameterization_2_sample_normal_sample_softplus_readvariableop_resource*&
_output_shapes
: *
dtype0�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log�kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2147908*
T0*
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOpiindependentnormal_constructed_at_conv2d_reparameterization_2_sample_normal_sample_readvariableop_resource*&
_output_shapes
: *
dtype0�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0�kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2147908*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv�kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2147908*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"�����������������
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * H[F�
truedivRealDiv�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0truediv/y:output:0*
T0*
_output_shapes
: K
divergence_kernelIdentitytruediv:z:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOptindependentnormal_constructed_at_conv2d_reparameterization_2_sample_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
: *
dtype0�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log�kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2147936*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOpkindependentnormal_constructed_at_conv2d_reparameterization_2_sample_1_normal_sample_readvariableop_resource*
_output_shapes
: *
dtype0�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0�kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2147936*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv�kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2147936*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 * H[F�
	truediv_1RealDiv�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0truediv_1/y:output:0*
T0*
_output_shapes
: K
divergence_biasIdentitytruediv_1:z:0*
T0*
_output_shapes
: k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:����������� Z

Identity_1Identitydivergence_kernel:output:0^NoOp*
T0*
_output_shapes
: X

Identity_2Identitydivergence_bias:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOpa^IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/ReadVariableOpj^IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOpc^IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/ReadVariableOpl^IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp�^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�^KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�^KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:�����������: : : : : : : : 2�
`IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/ReadVariableOp`IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/ReadVariableOp2�
iIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOpiIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOp2�
bIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/ReadVariableOpbIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/ReadVariableOp2�
kIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOpkIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp2�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_34_layer_call_and_return_conditional_losses_2148068

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@l
activation_16/SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:���������@
activation_16/mulMulactivation_16/Sigmoid:y:0BiasAdd:output:0*
T0*/
_output_shapes
:���������@l
IdentityIdentityactivation_16/mul:z:0^NoOp*
T0*/
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
N
2__inference_max_pooling2d_52_layer_call_fn_2150516

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_2147812�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
+__inference_conv2d_35_layer_call_fn_2150420

inputs"
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_35_layer_call_and_return_conditional_losses_2148101x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������		�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������		@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
e
,__inference_dropout_40_layer_call_fn_2150526

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_40_layer_call_and_return_conditional_losses_2148171x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
W__inference_dense_reparameterization_2_layer_call_and_return_conditional_losses_2148325

inputs{
hindependentnormal_constructed_at_dense_reparameterization_2_sample_normal_sample_readvariableop_resource:	��
qindependentnormal_constructed_at_dense_reparameterization_2_sample_normal_sample_softplus_readvariableop_resource:	�x
jindependentnormal_constructed_at_dense_reparameterization_2_sample_1_normal_sample_readvariableop_resource:�
sindependentnormal_constructed_at_dense_reparameterization_2_sample_1_normal_sample_softplus_readvariableop_resource:�
�kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2148266�
�kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�
�kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2148294�
�kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity

identity_1

identity_2��_IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/ReadVariableOp�hIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOp�aIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/ReadVariableOp�jIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp��KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp��KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp��KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp��KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�
OIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB �
]IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :�
nIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
_IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/ReadVariableOpReadVariableOphindependentnormal_constructed_at_dense_reparameterization_2_sample_normal_sample_readvariableop_resource*
_output_shapes
:	�*
dtype0�
hIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOpReadVariableOpqindependentnormal_constructed_at_dense_reparameterization_2_sample_normal_sample_softplus_readvariableop_resource*
_output_shapes
:	�*
dtype0�
YIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/SoftplusSoftpluspIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
VIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
TIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/addAddV2_IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/add/x:output:0gIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Softplus:activations:0*
T0*
_output_shapes
:	��
`IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"      �
VIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
dIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
fIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
fIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
^IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_sliceStridedSliceiIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/shape_as_tensor:output:0mIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice/stack:output:0oIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice/stack_1:output:0oIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
bIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"      �
XIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : �
fIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
hIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
hIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
`IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice_1StridedSlicekIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/shape_as_tensor_1:output:0oIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice_1/stack:output:0qIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice_1/stack_1:output:0qIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
aIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB �
cIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
^IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/BroadcastArgsBroadcastArgslIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/BroadcastArgs/s0_1:output:0gIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice:output:0*
_output_shapes
:�
`IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/BroadcastArgs_1BroadcastArgscIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/BroadcastArgs:r0:0iIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice_1:output:0*
_output_shapes
:�
`IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:�
\IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
WIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/concatConcatV2iIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/concat/values_0:output:0eIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/BroadcastArgs_1:r0:0eIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
jIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
zIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal`IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/concat:output:0*
T0*#
_output_shapes
:�*
dtype0�
iIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal/mulMul�IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal/RandomStandardNormal:output:0uIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal/stddev:output:0*
T0*#
_output_shapes
:��
eIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normalAddV2mIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal/mul:z:0sIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal/mean:output:0*
T0*#
_output_shapes
:��
TIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/mulMuliIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal:z:0XIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/add:z:0*
T0*#
_output_shapes
:��
VIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/add_1AddV2XIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/mul:z:0gIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/ReadVariableOp:value:0*
T0*#
_output_shapes
:��
^IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
XIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/ReshapeReshapeZIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/add_1:z:0gIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Reshape/shape:output:0*
T0*#
_output_shapes
:��
PIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
JIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/ReshapeReshapeaIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Reshape:output:0YIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Reshape/shape:output:0*
T0*
_output_shapes
:	��
MatMulMatMulinputsSIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Reshape:output:0*
T0*'
_output_shapes
:����������
QIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB �
_IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :�
pIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
aIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/ReadVariableOpReadVariableOpjindependentnormal_constructed_at_dense_reparameterization_2_sample_1_normal_sample_readvariableop_resource*
_output_shapes
:*
dtype0�
jIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOpReadVariableOpsindependentnormal_constructed_at_dense_reparameterization_2_sample_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
:*
dtype0�
[IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/SoftplusSoftplusrIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:�
XIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
VIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/addAddV2aIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/add/x:output:0iIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Softplus:activations:0*
T0*
_output_shapes
:�
bIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:�
XIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
fIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
hIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
hIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
`IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_sliceStridedSlicekIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/shape_as_tensor:output:0oIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice/stack:output:0qIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice/stack_1:output:0qIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
dIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:�
ZIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : �
hIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
jIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
jIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
bIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice_1StridedSlicemIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/shape_as_tensor_1:output:0qIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack:output:0sIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack_1:output:0sIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
cIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB �
eIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
`IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/BroadcastArgsBroadcastArgsnIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/BroadcastArgs/s0_1:output:0iIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice:output:0*
_output_shapes
:�
bIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/BroadcastArgs_1BroadcastArgseIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/BroadcastArgs:r0:0kIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:�
bIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:�
^IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
YIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/concatConcatV2kIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/concat/values_0:output:0gIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/BroadcastArgs_1:r0:0gIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
lIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    �
nIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
|IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalbIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/concat:output:0*
T0*
_output_shapes

:*
dtype0�
kIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal/mulMul�IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0wIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal/stddev:output:0*
T0*
_output_shapes

:�
gIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normalAddV2oIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal/mul:z:0uIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal/mean:output:0*
T0*
_output_shapes

:�
VIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/mulMulkIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal:z:0ZIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/add:z:0*
T0*
_output_shapes

:�
XIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/add_1AddV2ZIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/mul:z:0iIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/ReadVariableOp:value:0*
T0*
_output_shapes

:�
`IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
ZIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/ReshapeReshape\IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/add_1:z:0iIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

:�
RIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
LIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/ReshapeReshapecIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Reshape:output:0[IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Reshape/shape:output:0*
T0*
_output_shapes
:�
BiasAddBiasAddMatMul:product:0UIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Reshape:output:0*
T0*'
_output_shapes
:����������
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOpqindependentnormal_constructed_at_dense_reparameterization_2_sample_normal_sample_softplus_readvariableop_resource*
_output_shapes
:	�*
dtype0�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log�kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2148266*
T0*
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOphindependentnormal_constructed_at_dense_reparameterization_2_sample_normal_sample_readvariableop_resource*
_output_shapes
:	�*
dtype0�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0�kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2148266*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv�kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2148266*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"���������
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * H[F�
truedivRealDiv�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0truediv/y:output:0*
T0*
_output_shapes
: K
divergence_kernelIdentitytruediv:z:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOpsindependentnormal_constructed_at_dense_reparameterization_2_sample_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
:*
dtype0�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log�kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2148294*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOpjindependentnormal_constructed_at_dense_reparameterization_2_sample_1_normal_sample_readvariableop_resource*
_output_shapes
:*
dtype0�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0�kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2148294*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv�kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2148294*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 * H[F�
	truediv_1RealDiv�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0truediv_1/y:output:0*
T0*
_output_shapes
: K
divergence_biasIdentitytruediv_1:z:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identitydivergence_kernel:output:0^NoOp*
T0*
_output_shapes
: X

Identity_2Identitydivergence_bias:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp`^IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/ReadVariableOpi^IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOpb^IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/ReadVariableOpk^IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp�^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�^KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�^KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:����������: : : : : :	�: :2�
_IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/ReadVariableOp_IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/ReadVariableOp2�
hIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOphIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOp2�
aIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/ReadVariableOpaIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/ReadVariableOp2�
jIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOpjIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp2�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp: 

_output_shapes
::

_output_shapes
: :%!

_output_shapes
:	�:

_output_shapes
: :P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
,__inference_dropout_37_layer_call_fn_2150331

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_37_layer_call_and_return_conditional_losses_2148054w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
+__inference_conv2d_37_layer_call_fn_2150499

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_37_layer_call_and_return_conditional_losses_2148152x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
i
M__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_2150235

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
i
M__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_2150442

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
N
2__inference_max_pooling2d_49_layer_call_fn_2150321

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_2147776�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�f
�
J__inference_sequential_14_layer_call_and_return_conditional_losses_2148492%
!conv2d_reparameterization_2_input=
#conv2d_reparameterization_2_2148368: =
#conv2d_reparameterization_2_2148370: 1
#conv2d_reparameterization_2_2148372: 1
#conv2d_reparameterization_2_2148374: '
#conv2d_reparameterization_2_2148376'
#conv2d_reparameterization_2_2148378'
#conv2d_reparameterization_2_2148380'
#conv2d_reparameterization_2_21483825
depthwise_conv2d_13_2148388: *
depthwise_conv2d_13_2148390:	�,
conv2d_33_2148400:�@
conv2d_33_2148402:@+
conv2d_34_2148412:@@
conv2d_34_2148414:@,
conv2d_35_2148424:@� 
conv2d_35_2148426:	�-
conv2d_36_2148436:�� 
conv2d_36_2148438:	�-
conv2d_37_2148441:�� 
conv2d_37_2148443:	�5
"dense_reparameterization_2_2148454:	�5
"dense_reparameterization_2_2148456:	�0
"dense_reparameterization_2_2148458:0
"dense_reparameterization_2_2148460:&
"dense_reparameterization_2_2148462&
"dense_reparameterization_2_2148464&
"dense_reparameterization_2_2148466&
"dense_reparameterization_2_2148468
identity

identity_1

identity_2

identity_3

identity_4��!conv2d_33/StatefulPartitionedCall�!conv2d_34/StatefulPartitionedCall�!conv2d_35/StatefulPartitionedCall�!conv2d_36/StatefulPartitionedCall�!conv2d_37/StatefulPartitionedCall�3conv2d_reparameterization_2/StatefulPartitionedCall�2dense_reparameterization_2/StatefulPartitionedCall�+depthwise_conv2d_13/StatefulPartitionedCall�
3conv2d_reparameterization_2/StatefulPartitionedCallStatefulPartitionedCall!conv2d_reparameterization_2_input#conv2d_reparameterization_2_2148368#conv2d_reparameterization_2_2148370#conv2d_reparameterization_2_2148372#conv2d_reparameterization_2_2148374#conv2d_reparameterization_2_2148376#conv2d_reparameterization_2_2148378#conv2d_reparameterization_2_2148380#conv2d_reparameterization_2_2148382*
Tin
2	*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:����������� : : *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *a
f\RZ
X__inference_conv2d_reparameterization_2_layer_call_and_return_conditional_losses_2147967�
 max_pooling2d_47/PartitionedCallPartitionedCall<conv2d_reparameterization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������KK * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_2147752�
+depthwise_conv2d_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_47/PartitionedCall:output:0depthwise_conv2d_13_2148388depthwise_conv2d_13_2148390*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������KK�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_depthwise_conv2d_13_layer_call_and_return_conditional_losses_2148002�
 max_pooling2d_48/PartitionedCallPartitionedCall4depthwise_conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_2147764�
dropout_36/PartitionedCallPartitionedCall)max_pooling2d_48/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������%%�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_36_layer_call_and_return_conditional_losses_2148398�
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall#dropout_36/PartitionedCall:output:0conv2d_33_2148400conv2d_33_2148402*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������%%@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_33_layer_call_and_return_conditional_losses_2148035�
 max_pooling2d_49/PartitionedCallPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_2147776�
dropout_37/PartitionedCallPartitionedCall)max_pooling2d_49/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_37_layer_call_and_return_conditional_losses_2148410�
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall#dropout_37/PartitionedCall:output:0conv2d_34_2148412conv2d_34_2148414*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_34_layer_call_and_return_conditional_losses_2148068�
 max_pooling2d_50/PartitionedCallPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_2147788�
dropout_38/PartitionedCallPartitionedCall)max_pooling2d_50/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_38_layer_call_and_return_conditional_losses_2148422�
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall#dropout_38/PartitionedCall:output:0conv2d_35_2148424conv2d_35_2148426*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_35_layer_call_and_return_conditional_losses_2148101�
 max_pooling2d_51/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_2147800�
dropout_39/PartitionedCallPartitionedCall)max_pooling2d_51/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_39_layer_call_and_return_conditional_losses_2148434�
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall#dropout_39/PartitionedCall:output:0conv2d_36_2148436conv2d_36_2148438*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_36_layer_call_and_return_conditional_losses_2148134�
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0conv2d_37_2148441conv2d_37_2148443*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_37_layer_call_and_return_conditional_losses_2148152�
 max_pooling2d_52/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_2147812�
dropout_40/PartitionedCallPartitionedCall)max_pooling2d_52/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_40_layer_call_and_return_conditional_losses_2148451�
flatten_7/PartitionedCallPartitionedCall#dropout_40/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_7_layer_call_and_return_conditional_losses_2148179�
2dense_reparameterization_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0"dense_reparameterization_2_2148454"dense_reparameterization_2_2148456"dense_reparameterization_2_2148458"dense_reparameterization_2_2148460"dense_reparameterization_2_2148462"dense_reparameterization_2_2148464"dense_reparameterization_2_2148466"dense_reparameterization_2_2148468*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������: : *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *`
f[RY
W__inference_dense_reparameterization_2_layer_call_and_return_conditional_losses_2148325�
%one_hot_categorical_7/PartitionedCallPartitionedCall;dense_reparameterization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_one_hot_categorical_7_layer_call_and_return_conditional_losses_2148484}
IdentityIdentity.one_hot_categorical_7/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������|

Identity_1Identity<conv2d_reparameterization_2/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: |

Identity_2Identity<conv2d_reparameterization_2/StatefulPartitionedCall:output:2^NoOp*
T0*
_output_shapes
: {

Identity_3Identity;dense_reparameterization_2/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: {

Identity_4Identity;dense_reparameterization_2/StatefulPartitionedCall:output:2^NoOp*
T0*
_output_shapes
: �
NoOpNoOp"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall4^conv2d_reparameterization_2/StatefulPartitionedCall3^dense_reparameterization_2/StatefulPartitionedCall,^depthwise_conv2d_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:�����������: : : : : : : : : : : : : : : : : : : : : : : : : :	�: :2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2j
3conv2d_reparameterization_2/StatefulPartitionedCall3conv2d_reparameterization_2/StatefulPartitionedCall2h
2dense_reparameterization_2/StatefulPartitionedCall2dense_reparameterization_2/StatefulPartitionedCall2Z
+depthwise_conv2d_13/StatefulPartitionedCall+depthwise_conv2d_13/StatefulPartitionedCall: 

_output_shapes
::

_output_shapes
: :%!

_output_shapes
:	�:

_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :t p
1
_output_shapes
:�����������
;
_user_specified_name#!conv2d_reparameterization_2_input
�
e
G__inference_dropout_37_layer_call_and_return_conditional_losses_2150353

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
N
2__inference_max_pooling2d_50_layer_call_fn_2150379

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_2147788�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
+__inference_conv2d_33_layer_call_fn_2150304

inputs"
unknown:�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������%%@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_33_layer_call_and_return_conditional_losses_2148035w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������%%@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������%%�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
��
�8
J__inference_sequential_14_layer_call_and_return_conditional_losses_2149702

inputst
Zconv2d_reparameterization_2_independentnormal_sample_normal_sample_readvariableop_resource: }
cconv2d_reparameterization_2_independentnormal_sample_normal_sample_softplus_readvariableop_resource: j
\conv2d_reparameterization_2_independentnormal_sample_1_normal_sample_readvariableop_resource: s
econv2d_reparameterization_2_independentnormal_sample_1_normal_sample_softplus_readvariableop_resource: �
�conv2d_reparameterization_2_kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149398�
�conv2d_reparameterization_2_kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�
�conv2d_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149426�
�conv2d_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xO
5depthwise_conv2d_13_depthwise_readvariableop_resource: B
3depthwise_conv2d_13_biasadd_readvariableop_resource:	�C
(conv2d_33_conv2d_readvariableop_resource:�@7
)conv2d_33_biasadd_readvariableop_resource:@B
(conv2d_34_conv2d_readvariableop_resource:@@7
)conv2d_34_biasadd_readvariableop_resource:@C
(conv2d_35_conv2d_readvariableop_resource:@�8
)conv2d_35_biasadd_readvariableop_resource:	�D
(conv2d_36_conv2d_readvariableop_resource:��8
)conv2d_36_biasadd_readvariableop_resource:	�D
(conv2d_37_conv2d_readvariableop_resource:��8
)conv2d_37_biasadd_readvariableop_resource:	�l
Ydense_reparameterization_2_independentnormal_sample_normal_sample_readvariableop_resource:	�u
bdense_reparameterization_2_independentnormal_sample_normal_sample_softplus_readvariableop_resource:	�i
[dense_reparameterization_2_independentnormal_sample_1_normal_sample_readvariableop_resource:r
ddense_reparameterization_2_independentnormal_sample_1_normal_sample_softplus_readvariableop_resource:�
�dense_reparameterization_2_kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149634�
�dense_reparameterization_2_kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�
�dense_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149662�
�dense_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity

identity_1

identity_2

identity_3

identity_4�� conv2d_33/BiasAdd/ReadVariableOp�conv2d_33/Conv2D/ReadVariableOp� conv2d_34/BiasAdd/ReadVariableOp�conv2d_34/Conv2D/ReadVariableOp� conv2d_35/BiasAdd/ReadVariableOp�conv2d_35/Conv2D/ReadVariableOp� conv2d_36/BiasAdd/ReadVariableOp�conv2d_36/Conv2D/ReadVariableOp� conv2d_37/BiasAdd/ReadVariableOp�conv2d_37/Conv2D/ReadVariableOp�Qconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/ReadVariableOp�Zconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus/ReadVariableOp�Sconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReadVariableOp�\conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus/ReadVariableOp��conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp��conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp��conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp��conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�Pdense_reparameterization_2/IndependentNormal/sample/Normal/sample/ReadVariableOp�Ydense_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus/ReadVariableOp�Rdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReadVariableOp�[dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus/ReadVariableOp��dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp��dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp��dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp��dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�*depthwise_conv2d_13/BiasAdd/ReadVariableOp�,depthwise_conv2d_13/depthwise/ReadVariableOp�
Aconv2d_reparameterization_2/IndependentNormal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB �
Oconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :�
`conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
Qconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/ReadVariableOpReadVariableOpZconv2d_reparameterization_2_independentnormal_sample_normal_sample_readvariableop_resource*&
_output_shapes
: *
dtype0�
Zconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus/ReadVariableOpReadVariableOpcconv2d_reparameterization_2_independentnormal_sample_normal_sample_softplus_readvariableop_resource*&
_output_shapes
: *
dtype0�
Kconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/SoftplusSoftplusbconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
Hconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
Fconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/addAddV2Qconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/add/x:output:0Yconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus:activations:0*
T0*&
_output_shapes
: �
Rconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"             �
Hconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
Vconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Xconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Xconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Pconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_sliceStridedSlice[conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/shape_as_tensor:output:0_conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice/stack:output:0aconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice/stack_1:output:0aconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Tconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
Jconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : �
Xconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Zconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Zconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Rconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1StridedSlice]conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/shape_as_tensor_1:output:0aconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1/stack:output:0cconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1/stack_1:output:0cconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Sconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB �
Uconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
Pconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgsBroadcastArgs^conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgs/s0_1:output:0Yconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice:output:0*
_output_shapes
:�
Rconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgs_1BroadcastArgsUconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgs:r0:0[conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1:output:0*
_output_shapes
:�
Rconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:�
Nconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Iconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/concatConcatV2[conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/concat/values_0:output:0Wconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgs_1:r0:0Wconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
\conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    �
^conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
lconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalRconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/concat:output:0*
T0**
_output_shapes
: *
dtype0�
[conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/mulMuluconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/RandomStandardNormal:output:0gconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/stddev:output:0*
T0**
_output_shapes
: �
Wconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normalAddV2_conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/mul:z:0econv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/mean:output:0*
T0**
_output_shapes
: �
Fconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/mulMul[conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal:z:0Jconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/add:z:0*
T0**
_output_shapes
: �
Hconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/add_1AddV2Jconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/mul:z:0Yconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/ReadVariableOp:value:0*
T0**
_output_shapes
: �
Pconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"                �
Jconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/ReshapeReshapeLconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/add_1:z:0Yconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/Reshape/shape:output:0*
T0**
_output_shapes
: �
Bconv2d_reparameterization_2/IndependentNormal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
<conv2d_reparameterization_2/IndependentNormal/sample/ReshapeReshapeSconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/Reshape:output:0Kconv2d_reparameterization_2/IndependentNormal/sample/Reshape/shape:output:0*
T0*&
_output_shapes
: �
"conv2d_reparameterization_2/Conv2DConv2DinputsEconv2d_reparameterization_2/IndependentNormal/sample/Reshape:output:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
�
Cconv2d_reparameterization_2/IndependentNormal/sample_1/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB �
Qconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :�
bconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
Sconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReadVariableOpReadVariableOp\conv2d_reparameterization_2_independentnormal_sample_1_normal_sample_readvariableop_resource*
_output_shapes
: *
dtype0�
\conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus/ReadVariableOpReadVariableOpeconv2d_reparameterization_2_independentnormal_sample_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
: *
dtype0�
Mconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/SoftplusSoftplusdconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
: �
Jconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
Hconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/addAddV2Sconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/add/x:output:0[conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus:activations:0*
T0*
_output_shapes
: �
Tconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: �
Jconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
Xconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Zconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Zconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Rconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_sliceStridedSlice]conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/shape_as_tensor:output:0aconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice/stack:output:0cconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice/stack_1:output:0cconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Vconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB: �
Lconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : �
Zconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
\conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
\conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Tconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1StridedSlice_conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/shape_as_tensor_1:output:0cconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1/stack:output:0econv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1/stack_1:output:0econv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Uconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB �
Wconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
Rconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgsBroadcastArgs`conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgs/s0_1:output:0[conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice:output:0*
_output_shapes
:�
Tconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgs_1BroadcastArgsWconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgs:r0:0]conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:�
Tconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:�
Pconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Kconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/concatConcatV2]conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/concat/values_0:output:0Yconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgs_1:r0:0Yconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
^conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    �
`conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
nconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalTconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/concat:output:0*
T0*
_output_shapes

: *
dtype0�
]conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/mulMulwconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0iconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/stddev:output:0*
T0*
_output_shapes

: �
Yconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normalAddV2aconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/mul:z:0gconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/mean:output:0*
T0*
_output_shapes

: �
Hconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/mulMul]conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal:z:0Lconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/add:z:0*
T0*
_output_shapes

: �
Jconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/add_1AddV2Lconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/mul:z:0[conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReadVariableOp:value:0*
T0*
_output_shapes

: �
Rconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       �
Lconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReshapeReshapeNconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/add_1:z:0[conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

: �
Dconv2d_reparameterization_2/IndependentNormal/sample_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB: �
>conv2d_reparameterization_2/IndependentNormal/sample_1/ReshapeReshapeUconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Reshape:output:0Mconv2d_reparameterization_2/IndependentNormal/sample_1/Reshape/shape:output:0*
T0*
_output_shapes
: �
#conv2d_reparameterization_2/BiasAddBiasAdd+conv2d_reparameterization_2/Conv2D:output:0Gconv2d_reparameterization_2/IndependentNormal/sample_1/Reshape:output:0*
T0*1
_output_shapes
:����������� �
 conv2d_reparameterization_2/ReluRelu,conv2d_reparameterization_2/BiasAdd:output:0*
T0*1
_output_shapes
:����������� �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOpcconv2d_reparameterization_2_independentnormal_sample_normal_sample_softplus_readvariableop_resource*&
_output_shapes
: *
dtype0�
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log�conv2d_reparameterization_2_kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149398*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOpZconv2d_reparameterization_2_independentnormal_sample_normal_sample_readvariableop_resource*&
_output_shapes
: *
dtype0�
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0�conv2d_reparameterization_2_kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149398*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv�conv2d_reparameterization_2_kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�conv2d_reparameterization_2_kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149398*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"�����������������
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: j
%conv2d_reparameterization_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * H[F�
#conv2d_reparameterization_2/truedivRealDiv�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0.conv2d_reparameterization_2/truediv/y:output:0*
T0*
_output_shapes
: �
-conv2d_reparameterization_2/divergence_kernelIdentity'conv2d_reparameterization_2/truediv:z:0*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOpeconv2d_reparameterization_2_independentnormal_sample_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
: *
dtype0�
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log�conv2d_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149426*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp\conv2d_reparameterization_2_independentnormal_sample_1_normal_sample_readvariableop_resource*
_output_shapes
: *
dtype0�
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0�conv2d_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149426*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv�conv2d_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�conv2d_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149426*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
: �
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: l
'conv2d_reparameterization_2/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 * H[F�
%conv2d_reparameterization_2/truediv_1RealDiv�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:00conv2d_reparameterization_2/truediv_1/y:output:0*
T0*
_output_shapes
: �
+conv2d_reparameterization_2/divergence_biasIdentity)conv2d_reparameterization_2/truediv_1:z:0*
T0*
_output_shapes
: �
max_pooling2d_47/MaxPoolMaxPool.conv2d_reparameterization_2/Relu:activations:0*/
_output_shapes
:���������KK *
ksize
*
paddingVALID*
strides
�
,depthwise_conv2d_13/depthwise/ReadVariableOpReadVariableOp5depthwise_conv2d_13_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype0|
#depthwise_conv2d_13/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             |
+depthwise_conv2d_13/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
depthwise_conv2d_13/depthwiseDepthwiseConv2dNative!max_pooling2d_47/MaxPool:output:04depthwise_conv2d_13/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
�
*depthwise_conv2d_13/BiasAdd/ReadVariableOpReadVariableOp3depthwise_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
depthwise_conv2d_13/BiasAddBiasAdd&depthwise_conv2d_13/depthwise:output:02depthwise_conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK��
)depthwise_conv2d_13/activation_16/SigmoidSigmoid$depthwise_conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK��
%depthwise_conv2d_13/activation_16/mulMul-depthwise_conv2d_13/activation_16/Sigmoid:y:0$depthwise_conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK��
max_pooling2d_48/MaxPoolMaxPool)depthwise_conv2d_13/activation_16/mul:z:0*0
_output_shapes
:���������%%�*
ksize
*
paddingVALID*
strides
]
dropout_36/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
dropout_36/dropout/MulMul!max_pooling2d_48/MaxPool:output:0!dropout_36/dropout/Const:output:0*
T0*0
_output_shapes
:���������%%�w
dropout_36/dropout/ShapeShape!max_pooling2d_48/MaxPool:output:0*
T0*
_output_shapes
::���
/dropout_36/dropout/random_uniform/RandomUniformRandomUniform!dropout_36/dropout/Shape:output:0*
T0*0
_output_shapes
:���������%%�*
dtype0f
!dropout_36/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout_36/dropout/GreaterEqualGreaterEqual8dropout_36/dropout/random_uniform/RandomUniform:output:0*dropout_36/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:���������%%�_
dropout_36/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_36/dropout/SelectV2SelectV2#dropout_36/dropout/GreaterEqual:z:0dropout_36/dropout/Mul:z:0#dropout_36/dropout/Const_1:output:0*
T0*0
_output_shapes
:���������%%��
conv2d_33/Conv2D/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
conv2d_33/Conv2DConv2D$dropout_36/dropout/SelectV2:output:0'conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������%%@*
paddingSAME*
strides
�
 conv2d_33/BiasAdd/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_33/BiasAddBiasAddconv2d_33/Conv2D:output:0(conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������%%@�
conv2d_33/activation_16/SigmoidSigmoidconv2d_33/BiasAdd:output:0*
T0*/
_output_shapes
:���������%%@�
conv2d_33/activation_16/mulMul#conv2d_33/activation_16/Sigmoid:y:0conv2d_33/BiasAdd:output:0*
T0*/
_output_shapes
:���������%%@�
max_pooling2d_49/MaxPoolMaxPoolconv2d_33/activation_16/mul:z:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
]
dropout_37/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
dropout_37/dropout/MulMul!max_pooling2d_49/MaxPool:output:0!dropout_37/dropout/Const:output:0*
T0*/
_output_shapes
:���������@w
dropout_37/dropout/ShapeShape!max_pooling2d_49/MaxPool:output:0*
T0*
_output_shapes
::���
/dropout_37/dropout/random_uniform/RandomUniformRandomUniform!dropout_37/dropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype0f
!dropout_37/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout_37/dropout/GreaterEqualGreaterEqual8dropout_37/dropout/random_uniform/RandomUniform:output:0*dropout_37/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@_
dropout_37/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_37/dropout/SelectV2SelectV2#dropout_37/dropout/GreaterEqual:z:0dropout_37/dropout/Mul:z:0#dropout_37/dropout/Const_1:output:0*
T0*/
_output_shapes
:���������@�
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_34/Conv2DConv2D$dropout_37/dropout/SelectV2:output:0'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
conv2d_34/activation_16/SigmoidSigmoidconv2d_34/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
conv2d_34/activation_16/mulMul#conv2d_34/activation_16/Sigmoid:y:0conv2d_34/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
max_pooling2d_50/MaxPoolMaxPoolconv2d_34/activation_16/mul:z:0*/
_output_shapes
:���������		@*
ksize
*
paddingVALID*
strides
]
dropout_38/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
dropout_38/dropout/MulMul!max_pooling2d_50/MaxPool:output:0!dropout_38/dropout/Const:output:0*
T0*/
_output_shapes
:���������		@w
dropout_38/dropout/ShapeShape!max_pooling2d_50/MaxPool:output:0*
T0*
_output_shapes
::���
/dropout_38/dropout/random_uniform/RandomUniformRandomUniform!dropout_38/dropout/Shape:output:0*
T0*/
_output_shapes
:���������		@*
dtype0f
!dropout_38/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout_38/dropout/GreaterEqualGreaterEqual8dropout_38/dropout/random_uniform/RandomUniform:output:0*dropout_38/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������		@_
dropout_38/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_38/dropout/SelectV2SelectV2#dropout_38/dropout/GreaterEqual:z:0dropout_38/dropout/Mul:z:0#dropout_38/dropout/Const_1:output:0*
T0*/
_output_shapes
:���������		@�
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_35/Conv2DConv2D$dropout_38/dropout/SelectV2:output:0'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
�
 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		��
conv2d_35/activation_16/SigmoidSigmoidconv2d_35/BiasAdd:output:0*
T0*0
_output_shapes
:���������		��
conv2d_35/activation_16/mulMul#conv2d_35/activation_16/Sigmoid:y:0conv2d_35/BiasAdd:output:0*
T0*0
_output_shapes
:���������		��
max_pooling2d_51/MaxPoolMaxPoolconv2d_35/activation_16/mul:z:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
]
dropout_39/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
dropout_39/dropout/MulMul!max_pooling2d_51/MaxPool:output:0!dropout_39/dropout/Const:output:0*
T0*0
_output_shapes
:����������w
dropout_39/dropout/ShapeShape!max_pooling2d_51/MaxPool:output:0*
T0*
_output_shapes
::���
/dropout_39/dropout/random_uniform/RandomUniformRandomUniform!dropout_39/dropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype0f
!dropout_39/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout_39/dropout/GreaterEqualGreaterEqual8dropout_39/dropout/random_uniform/RandomUniform:output:0*dropout_39/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������_
dropout_39/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_39/dropout/SelectV2SelectV2#dropout_39/dropout/GreaterEqual:z:0dropout_39/dropout/Mul:z:0#dropout_39/dropout/Const_1:output:0*
T0*0
_output_shapes
:�����������
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_36/Conv2DConv2D$dropout_39/dropout/SelectV2:output:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
conv2d_36/activation_16/SigmoidSigmoidconv2d_36/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
conv2d_36/activation_16/mulMul#conv2d_36/activation_16/Sigmoid:y:0conv2d_36/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_37/Conv2DConv2Dconv2d_36/activation_16/mul:z:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
conv2d_37/activation_16/SigmoidSigmoidconv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
conv2d_37/activation_16/mulMul#conv2d_37/activation_16/Sigmoid:y:0conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
max_pooling2d_52/MaxPoolMaxPoolconv2d_37/activation_16/mul:z:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
]
dropout_40/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
dropout_40/dropout/MulMul!max_pooling2d_52/MaxPool:output:0!dropout_40/dropout/Const:output:0*
T0*0
_output_shapes
:����������w
dropout_40/dropout/ShapeShape!max_pooling2d_52/MaxPool:output:0*
T0*
_output_shapes
::���
/dropout_40/dropout/random_uniform/RandomUniformRandomUniform!dropout_40/dropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype0f
!dropout_40/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout_40/dropout/GreaterEqualGreaterEqual8dropout_40/dropout/random_uniform/RandomUniform:output:0*dropout_40/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������_
dropout_40/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_40/dropout/SelectV2SelectV2#dropout_40/dropout/GreaterEqual:z:0dropout_40/dropout/Mul:z:0#dropout_40/dropout/Const_1:output:0*
T0*0
_output_shapes
:����������`
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten_7/ReshapeReshape$dropout_40/dropout/SelectV2:output:0flatten_7/Const:output:0*
T0*(
_output_shapes
:�����������
@dense_reparameterization_2/IndependentNormal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB �
Ndense_reparameterization_2/IndependentNormal/sample/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :�
_dense_reparameterization_2/IndependentNormal/sample/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
Pdense_reparameterization_2/IndependentNormal/sample/Normal/sample/ReadVariableOpReadVariableOpYdense_reparameterization_2_independentnormal_sample_normal_sample_readvariableop_resource*
_output_shapes
:	�*
dtype0�
Ydense_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus/ReadVariableOpReadVariableOpbdense_reparameterization_2_independentnormal_sample_normal_sample_softplus_readvariableop_resource*
_output_shapes
:	�*
dtype0�
Jdense_reparameterization_2/IndependentNormal/sample/Normal/sample/SoftplusSoftplusadense_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
Gdense_reparameterization_2/IndependentNormal/sample/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
Edense_reparameterization_2/IndependentNormal/sample/Normal/sample/addAddV2Pdense_reparameterization_2/IndependentNormal/sample/Normal/sample/add/x:output:0Xdense_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus:activations:0*
T0*
_output_shapes
:	��
Qdense_reparameterization_2/IndependentNormal/sample/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"      �
Gdense_reparameterization_2/IndependentNormal/sample/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
Udense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Wdense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Wdense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Odense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_sliceStridedSliceZdense_reparameterization_2/IndependentNormal/sample/Normal/sample/shape_as_tensor:output:0^dense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice/stack:output:0`dense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice/stack_1:output:0`dense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Sdense_reparameterization_2/IndependentNormal/sample/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"      �
Idense_reparameterization_2/IndependentNormal/sample/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : �
Wdense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Ydense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Ydense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Qdense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1StridedSlice\dense_reparameterization_2/IndependentNormal/sample/Normal/sample/shape_as_tensor_1:output:0`dense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1/stack:output:0bdense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1/stack_1:output:0bdense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Rdense_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB �
Tdense_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
Odense_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgsBroadcastArgs]dense_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgs/s0_1:output:0Xdense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice:output:0*
_output_shapes
:�
Qdense_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgs_1BroadcastArgsTdense_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgs:r0:0Zdense_reparameterization_2/IndependentNormal/sample/Normal/sample/strided_slice_1:output:0*
_output_shapes
:�
Qdense_reparameterization_2/IndependentNormal/sample/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:�
Mdense_reparameterization_2/IndependentNormal/sample/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Hdense_reparameterization_2/IndependentNormal/sample/Normal/sample/concatConcatV2Zdense_reparameterization_2/IndependentNormal/sample/Normal/sample/concat/values_0:output:0Vdense_reparameterization_2/IndependentNormal/sample/Normal/sample/BroadcastArgs_1:r0:0Vdense_reparameterization_2/IndependentNormal/sample/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
[dense_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    �
]dense_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
kdense_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalQdense_reparameterization_2/IndependentNormal/sample/Normal/sample/concat:output:0*
T0*#
_output_shapes
:�*
dtype0�
Zdense_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/mulMultdense_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/RandomStandardNormal:output:0fdense_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/stddev:output:0*
T0*#
_output_shapes
:��
Vdense_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normalAddV2^dense_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/mul:z:0ddense_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal/mean:output:0*
T0*#
_output_shapes
:��
Edense_reparameterization_2/IndependentNormal/sample/Normal/sample/mulMulZdense_reparameterization_2/IndependentNormal/sample/Normal/sample/normal/random_normal:z:0Idense_reparameterization_2/IndependentNormal/sample/Normal/sample/add:z:0*
T0*#
_output_shapes
:��
Gdense_reparameterization_2/IndependentNormal/sample/Normal/sample/add_1AddV2Idense_reparameterization_2/IndependentNormal/sample/Normal/sample/mul:z:0Xdense_reparameterization_2/IndependentNormal/sample/Normal/sample/ReadVariableOp:value:0*
T0*#
_output_shapes
:��
Odense_reparameterization_2/IndependentNormal/sample/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
Idense_reparameterization_2/IndependentNormal/sample/Normal/sample/ReshapeReshapeKdense_reparameterization_2/IndependentNormal/sample/Normal/sample/add_1:z:0Xdense_reparameterization_2/IndependentNormal/sample/Normal/sample/Reshape/shape:output:0*
T0*#
_output_shapes
:��
Adense_reparameterization_2/IndependentNormal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
;dense_reparameterization_2/IndependentNormal/sample/ReshapeReshapeRdense_reparameterization_2/IndependentNormal/sample/Normal/sample/Reshape:output:0Jdense_reparameterization_2/IndependentNormal/sample/Reshape/shape:output:0*
T0*
_output_shapes
:	��
!dense_reparameterization_2/MatMulMatMulflatten_7/Reshape:output:0Ddense_reparameterization_2/IndependentNormal/sample/Reshape:output:0*
T0*'
_output_shapes
:����������
Bdense_reparameterization_2/IndependentNormal/sample_1/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB �
Pdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :�
adense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
Rdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReadVariableOpReadVariableOp[dense_reparameterization_2_independentnormal_sample_1_normal_sample_readvariableop_resource*
_output_shapes
:*
dtype0�
[dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus/ReadVariableOpReadVariableOpddense_reparameterization_2_independentnormal_sample_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
:*
dtype0�
Ldense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/SoftplusSoftpluscdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:�
Idense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
Gdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/addAddV2Rdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/add/x:output:0Zdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus:activations:0*
T0*
_output_shapes
:�
Sdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:�
Idense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
Wdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Ydense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Ydense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Qdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_sliceStridedSlice\dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/shape_as_tensor:output:0`dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice/stack:output:0bdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice/stack_1:output:0bdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Udense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:�
Kdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : �
Ydense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
[dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
[dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Sdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1StridedSlice^dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/shape_as_tensor_1:output:0bdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1/stack:output:0ddense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1/stack_1:output:0ddense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
Tdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB �
Vdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
Qdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgsBroadcastArgs_dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgs/s0_1:output:0Zdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice:output:0*
_output_shapes
:�
Sdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgs_1BroadcastArgsVdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgs:r0:0\dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:�
Sdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:�
Odense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Jdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/concatConcatV2\dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/concat/values_0:output:0Xdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/BroadcastArgs_1:r0:0Xdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
]dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    �
_dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
mdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalSdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/concat:output:0*
T0*
_output_shapes

:*
dtype0�
\dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/mulMulvdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0hdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/stddev:output:0*
T0*
_output_shapes

:�
Xdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normalAddV2`dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/mul:z:0fdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal/mean:output:0*
T0*
_output_shapes

:�
Gdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/mulMul\dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/normal/random_normal:z:0Kdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/add:z:0*
T0*
_output_shapes

:�
Idense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/add_1AddV2Kdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/mul:z:0Zdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReadVariableOp:value:0*
T0*
_output_shapes

:�
Qdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
Kdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReshapeReshapeMdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/add_1:z:0Zdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

:�
Cdense_reparameterization_2/IndependentNormal/sample_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
=dense_reparameterization_2/IndependentNormal/sample_1/ReshapeReshapeTdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Reshape:output:0Ldense_reparameterization_2/IndependentNormal/sample_1/Reshape/shape:output:0*
T0*
_output_shapes
:�
"dense_reparameterization_2/BiasAddBiasAdd+dense_reparameterization_2/MatMul:product:0Fdense_reparameterization_2/IndependentNormal/sample_1/Reshape:output:0*
T0*'
_output_shapes
:����������
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOpbdense_reparameterization_2_independentnormal_sample_normal_sample_softplus_readvariableop_resource*
_output_shapes
:	�*
dtype0�
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log�dense_reparameterization_2_kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149634*
T0*
_output_shapes
: �
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOpYdense_reparameterization_2_independentnormal_sample_normal_sample_readvariableop_resource*
_output_shapes
:	�*
dtype0�
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0�dense_reparameterization_2_kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149634*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv�dense_reparameterization_2_kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�dense_reparameterization_2_kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149634*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:	��
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"���������
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: i
$dense_reparameterization_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * H[F�
"dense_reparameterization_2/truedivRealDiv�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0-dense_reparameterization_2/truediv/y:output:0*
T0*
_output_shapes
: �
,dense_reparameterization_2/divergence_kernelIdentity&dense_reparameterization_2/truediv:z:0*
T0*
_output_shapes
: �
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOpddense_reparameterization_2_independentnormal_sample_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
:*
dtype0�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log�dense_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149662*
T0*
_output_shapes
: �
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp[dense_reparameterization_2_independentnormal_sample_1_normal_sample_readvariableop_resource*
_output_shapes
:*
dtype0�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0�dense_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149662*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv�dense_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�dense_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2149662*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: k
&dense_reparameterization_2/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 * H[F�
$dense_reparameterization_2/truediv_1RealDiv�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0/dense_reparameterization_2/truediv_1/y:output:0*
T0*
_output_shapes
: �
*dense_reparameterization_2/divergence_biasIdentity(dense_reparameterization_2/truediv_1:z:0*
T0*
_output_shapes
: �
Lone_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/IdentityIdentity+dense_reparameterization_2/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Tone_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
����������
Jone_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/ArgMaxArgMaxUone_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/Identity:output:0]one_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:����������
Tone_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Uone_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Qone_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :�
Kone_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/one_hotOneHotSone_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/ArgMax:output:0Zone_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/one_hot/depth:output:0]one_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/one_hot/on_value:output:0^one_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/one_hot/off_value:output:0*
T0*'
_output_shapes
:����������
IdentityIdentityTone_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/one_hot:output:0^NoOp*
T0*'
_output_shapes
:���������v

Identity_1Identity6conv2d_reparameterization_2/divergence_kernel:output:0^NoOp*
T0*
_output_shapes
: t

Identity_2Identity4conv2d_reparameterization_2/divergence_bias:output:0^NoOp*
T0*
_output_shapes
: u

Identity_3Identity5dense_reparameterization_2/divergence_kernel:output:0^NoOp*
T0*
_output_shapes
: s

Identity_4Identity3dense_reparameterization_2/divergence_bias:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp!^conv2d_33/BiasAdd/ReadVariableOp ^conv2d_33/Conv2D/ReadVariableOp!^conv2d_34/BiasAdd/ReadVariableOp ^conv2d_34/Conv2D/ReadVariableOp!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOpR^conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/ReadVariableOp[^conv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus/ReadVariableOpT^conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReadVariableOp]^conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus/ReadVariableOp�^conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�^conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�^conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�^conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpQ^dense_reparameterization_2/IndependentNormal/sample/Normal/sample/ReadVariableOpZ^dense_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus/ReadVariableOpS^dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReadVariableOp\^dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus/ReadVariableOp�^dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�^dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�^dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�^dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp+^depthwise_conv2d_13/BiasAdd/ReadVariableOp-^depthwise_conv2d_13/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:�����������: : : : : : : : : : : : : : : : : : : : : : : : : :	�: :2D
 conv2d_33/BiasAdd/ReadVariableOp conv2d_33/BiasAdd/ReadVariableOp2B
conv2d_33/Conv2D/ReadVariableOpconv2d_33/Conv2D/ReadVariableOp2D
 conv2d_34/BiasAdd/ReadVariableOp conv2d_34/BiasAdd/ReadVariableOp2B
conv2d_34/Conv2D/ReadVariableOpconv2d_34/Conv2D/ReadVariableOp2D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2D
 conv2d_36/BiasAdd/ReadVariableOp conv2d_36/BiasAdd/ReadVariableOp2B
conv2d_36/Conv2D/ReadVariableOpconv2d_36/Conv2D/ReadVariableOp2D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp2�
Qconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/ReadVariableOpQconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/ReadVariableOp2�
Zconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus/ReadVariableOpZconv2d_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus/ReadVariableOp2�
Sconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReadVariableOpSconv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReadVariableOp2�
\conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus/ReadVariableOp\conv2d_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus/ReadVariableOp2�
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2�
�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2�
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2�
�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2�
Pdense_reparameterization_2/IndependentNormal/sample/Normal/sample/ReadVariableOpPdense_reparameterization_2/IndependentNormal/sample/Normal/sample/ReadVariableOp2�
Ydense_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus/ReadVariableOpYdense_reparameterization_2/IndependentNormal/sample/Normal/sample/Softplus/ReadVariableOp2�
Rdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReadVariableOpRdense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/ReadVariableOp2�
[dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus/ReadVariableOp[dense_reparameterization_2/IndependentNormal/sample_1/Normal/sample/Softplus/ReadVariableOp2�
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2�
�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2�
�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2X
*depthwise_conv2d_13/BiasAdd/ReadVariableOp*depthwise_conv2d_13/BiasAdd/ReadVariableOp2\
,depthwise_conv2d_13/depthwise/ReadVariableOp,depthwise_conv2d_13/depthwise/ReadVariableOp: 

_output_shapes
::

_output_shapes
: :%!

_output_shapes
:	�:

_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
b
F__inference_flatten_7_layer_call_and_return_conditional_losses_2148179

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
N
2__inference_max_pooling2d_51_layer_call_fn_2150437

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_2147800�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
i
M__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_2147812

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
e
,__inference_dropout_39_layer_call_fn_2150447

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_39_layer_call_and_return_conditional_losses_2148120x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
��
�C
"__inference__wrapped_model_2147746%
!conv2d_reparameterization_2_input�
�sequential_14_conv2d_reparameterization_2_independentnormal_constructed_at_conv2d_reparameterization_2_sample_normal_sample_readvariableop_resource: �
�sequential_14_conv2d_reparameterization_2_independentnormal_constructed_at_conv2d_reparameterization_2_sample_normal_sample_softplus_readvariableop_resource: �
�sequential_14_conv2d_reparameterization_2_independentnormal_constructed_at_conv2d_reparameterization_2_sample_1_normal_sample_readvariableop_resource: �
�sequential_14_conv2d_reparameterization_2_independentnormal_constructed_at_conv2d_reparameterization_2_sample_1_normal_sample_softplus_readvariableop_resource: �
�sequential_14_conv2d_reparameterization_2_kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2147481�
�sequential_14_conv2d_reparameterization_2_kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�
�sequential_14_conv2d_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2147509�
�sequential_14_conv2d_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x]
Csequential_14_depthwise_conv2d_13_depthwise_readvariableop_resource: P
Asequential_14_depthwise_conv2d_13_biasadd_readvariableop_resource:	�Q
6sequential_14_conv2d_33_conv2d_readvariableop_resource:�@E
7sequential_14_conv2d_33_biasadd_readvariableop_resource:@P
6sequential_14_conv2d_34_conv2d_readvariableop_resource:@@E
7sequential_14_conv2d_34_biasadd_readvariableop_resource:@Q
6sequential_14_conv2d_35_conv2d_readvariableop_resource:@�F
7sequential_14_conv2d_35_biasadd_readvariableop_resource:	�R
6sequential_14_conv2d_36_conv2d_readvariableop_resource:��F
7sequential_14_conv2d_36_biasadd_readvariableop_resource:	�R
6sequential_14_conv2d_37_conv2d_readvariableop_resource:��F
7sequential_14_conv2d_37_biasadd_readvariableop_resource:	��
�sequential_14_dense_reparameterization_2_independentnormal_constructed_at_dense_reparameterization_2_sample_normal_sample_readvariableop_resource:	��
�sequential_14_dense_reparameterization_2_independentnormal_constructed_at_dense_reparameterization_2_sample_normal_sample_softplus_readvariableop_resource:	��
�sequential_14_dense_reparameterization_2_independentnormal_constructed_at_dense_reparameterization_2_sample_1_normal_sample_readvariableop_resource:�
�sequential_14_dense_reparameterization_2_independentnormal_constructed_at_dense_reparameterization_2_sample_1_normal_sample_softplus_readvariableop_resource:�
�sequential_14_dense_reparameterization_2_kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2147682�
�sequential_14_dense_reparameterization_2_kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�
�sequential_14_dense_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2147710�
�sequential_14_dense_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity��.sequential_14/conv2d_33/BiasAdd/ReadVariableOp�-sequential_14/conv2d_33/Conv2D/ReadVariableOp�.sequential_14/conv2d_34/BiasAdd/ReadVariableOp�-sequential_14/conv2d_34/Conv2D/ReadVariableOp�.sequential_14/conv2d_35/BiasAdd/ReadVariableOp�-sequential_14/conv2d_35/Conv2D/ReadVariableOp�.sequential_14/conv2d_36/BiasAdd/ReadVariableOp�-sequential_14/conv2d_36/Conv2D/ReadVariableOp�.sequential_14/conv2d_37/BiasAdd/ReadVariableOp�-sequential_14/conv2d_37/Conv2D/ReadVariableOp��sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/ReadVariableOp��sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOp��sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/ReadVariableOp��sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp��sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp��sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp��sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp��sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp��sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/ReadVariableOp��sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOp��sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/ReadVariableOp��sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp��sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp��sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp��sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp��sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�8sequential_14/depthwise_conv2d_13/BiasAdd/ReadVariableOp�:sequential_14/depthwise_conv2d_13/depthwise/ReadVariableOp�
zsequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/ReadVariableOpReadVariableOp�sequential_14_conv2d_reparameterization_2_independentnormal_constructed_at_conv2d_reparameterization_2_sample_normal_sample_readvariableop_resource*&
_output_shapes
: *
dtype0�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOpReadVariableOp�sequential_14_conv2d_reparameterization_2_independentnormal_constructed_at_conv2d_reparameterization_2_sample_normal_sample_softplus_readvariableop_resource*&
_output_shapes
: *
dtype0�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/SoftplusSoftplus�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/addAddV2�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/add/x:output:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Softplus:activations:0*
T0*&
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"             �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_sliceStridedSlice�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/shape_as_tensor:output:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice/stack:output:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice/stack_1:output:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice_1StridedSlice�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/shape_as_tensor_1:output:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice_1/stack:output:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice_1/stack_1:output:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/BroadcastArgsBroadcastArgs�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/BroadcastArgs/s0_1:output:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice:output:0*
_output_shapes
:�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/BroadcastArgs_1BroadcastArgs�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/BroadcastArgs:r0:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice_1:output:0*
_output_shapes
:�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/concatConcatV2�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/concat/values_0:output:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/BroadcastArgs_1:r0:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/concat:output:0*
T0**
_output_shapes
: *
dtype0�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal/mulMul�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal/RandomStandardNormal:output:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal/stddev:output:0*
T0**
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normalAddV2�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal/mul:z:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal/mean:output:0*
T0**
_output_shapes
: �
sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/mulMul�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal:z:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/add:z:0*
T0**
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/add_1AddV2�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/mul:z:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/ReadVariableOp:value:0*
T0**
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"                �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/ReshapeReshape�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/add_1:z:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Reshape/shape:output:0*
T0**
_output_shapes
: �
{sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
usequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/ReshapeReshape�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Reshape:output:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Reshape/shape:output:0*
T0*&
_output_shapes
: �
0sequential_14/conv2d_reparameterization_2/Conv2DConv2D!conv2d_reparameterization_2_input~sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Reshape:output:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
�
|sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/ReadVariableOpReadVariableOp�sequential_14_conv2d_reparameterization_2_independentnormal_constructed_at_conv2d_reparameterization_2_sample_1_normal_sample_readvariableop_resource*
_output_shapes
: *
dtype0�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOpReadVariableOp�sequential_14_conv2d_reparameterization_2_independentnormal_constructed_at_conv2d_reparameterization_2_sample_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
: *
dtype0�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/SoftplusSoftplus�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/addAddV2�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/add/x:output:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Softplus:activations:0*
T0*
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_sliceStridedSlice�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/shape_as_tensor:output:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice/stack:output:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice/stack_1:output:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB: �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice_1StridedSlice�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/shape_as_tensor_1:output:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack:output:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack_1:output:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/BroadcastArgsBroadcastArgs�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/BroadcastArgs/s0_1:output:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice:output:0*
_output_shapes
:�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/BroadcastArgs_1BroadcastArgs�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/BroadcastArgs:r0:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/concatConcatV2�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/concat/values_0:output:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/BroadcastArgs_1:r0:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/concat:output:0*
T0*
_output_shapes

: *
dtype0�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal/mulMul�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal/stddev:output:0*
T0*
_output_shapes

: �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normalAddV2�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal/mul:z:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal/mean:output:0*
T0*
_output_shapes

: �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/mulMul�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal:z:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/add:z:0*
T0*
_output_shapes

: �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/add_1AddV2�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/mul:z:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/ReadVariableOp:value:0*
T0*
_output_shapes

: �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       �
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/ReshapeReshape�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/add_1:z:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

: �
}sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB: �
wsequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/ReshapeReshape�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Reshape:output:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Reshape/shape:output:0*
T0*
_output_shapes
: �
1sequential_14/conv2d_reparameterization_2/BiasAddBiasAdd9sequential_14/conv2d_reparameterization_2/Conv2D:output:0�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Reshape:output:0*
T0*1
_output_shapes
:����������� �
.sequential_14/conv2d_reparameterization_2/ReluRelu:sequential_14/conv2d_reparameterization_2/BiasAdd:output:0*
T0*1
_output_shapes
:����������� �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp�sequential_14_conv2d_reparameterization_2_independentnormal_constructed_at_conv2d_reparameterization_2_sample_normal_sample_softplus_readvariableop_resource*&
_output_shapes
: *
dtype0�
�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*&
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*&
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log�sequential_14_conv2d_reparameterization_2_kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2147481*
T0*
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*&
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp�sequential_14_conv2d_reparameterization_2_independentnormal_constructed_at_conv2d_reparameterization_2_sample_normal_sample_readvariableop_resource*&
_output_shapes
: *
dtype0�
�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0�sequential_14_conv2d_reparameterization_2_kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2147481*
T0*&
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv�sequential_14_conv2d_reparameterization_2_kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�sequential_14_conv2d_reparameterization_2_kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2147481*
T0*&
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*&
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*&
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*&
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*&
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*&
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"�����������������
�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: x
3sequential_14/conv2d_reparameterization_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * H[F�
1sequential_14/conv2d_reparameterization_2/truedivRealDiv�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0<sequential_14/conv2d_reparameterization_2/truediv/y:output:0*
T0*
_output_shapes
: �
;sequential_14/conv2d_reparameterization_2/divergence_kernelIdentity5sequential_14/conv2d_reparameterization_2/truediv:z:0*
T0*
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp�sequential_14_conv2d_reparameterization_2_independentnormal_constructed_at_conv2d_reparameterization_2_sample_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
: *
dtype0�
�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log�sequential_14_conv2d_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2147509*
T0*
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp�sequential_14_conv2d_reparameterization_2_independentnormal_constructed_at_conv2d_reparameterization_2_sample_1_normal_sample_readvariableop_resource*
_output_shapes
: *
dtype0�
�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0�sequential_14_conv2d_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2147509*
T0*
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv�sequential_14_conv2d_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�sequential_14_conv2d_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2147509*
T0*
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
: �
�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: z
5sequential_14/conv2d_reparameterization_2/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 * H[F�
3sequential_14/conv2d_reparameterization_2/truediv_1RealDiv�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0>sequential_14/conv2d_reparameterization_2/truediv_1/y:output:0*
T0*
_output_shapes
: �
9sequential_14/conv2d_reparameterization_2/divergence_biasIdentity7sequential_14/conv2d_reparameterization_2/truediv_1:z:0*
T0*
_output_shapes
: �
&sequential_14/max_pooling2d_47/MaxPoolMaxPool<sequential_14/conv2d_reparameterization_2/Relu:activations:0*/
_output_shapes
:���������KK *
ksize
*
paddingVALID*
strides
�
:sequential_14/depthwise_conv2d_13/depthwise/ReadVariableOpReadVariableOpCsequential_14_depthwise_conv2d_13_depthwise_readvariableop_resource*&
_output_shapes
: *
dtype0�
1sequential_14/depthwise_conv2d_13/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
9sequential_14/depthwise_conv2d_13/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
+sequential_14/depthwise_conv2d_13/depthwiseDepthwiseConv2dNative/sequential_14/max_pooling2d_47/MaxPool:output:0Bsequential_14/depthwise_conv2d_13/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
�
8sequential_14/depthwise_conv2d_13/BiasAdd/ReadVariableOpReadVariableOpAsequential_14_depthwise_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)sequential_14/depthwise_conv2d_13/BiasAddBiasAdd4sequential_14/depthwise_conv2d_13/depthwise:output:0@sequential_14/depthwise_conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK��
7sequential_14/depthwise_conv2d_13/activation_16/SigmoidSigmoid2sequential_14/depthwise_conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK��
3sequential_14/depthwise_conv2d_13/activation_16/mulMul;sequential_14/depthwise_conv2d_13/activation_16/Sigmoid:y:02sequential_14/depthwise_conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:���������KK��
&sequential_14/max_pooling2d_48/MaxPoolMaxPool7sequential_14/depthwise_conv2d_13/activation_16/mul:z:0*0
_output_shapes
:���������%%�*
ksize
*
paddingVALID*
strides
�
!sequential_14/dropout_36/IdentityIdentity/sequential_14/max_pooling2d_48/MaxPool:output:0*
T0*0
_output_shapes
:���������%%��
-sequential_14/conv2d_33/Conv2D/ReadVariableOpReadVariableOp6sequential_14_conv2d_33_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
sequential_14/conv2d_33/Conv2DConv2D*sequential_14/dropout_36/Identity:output:05sequential_14/conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������%%@*
paddingSAME*
strides
�
.sequential_14/conv2d_33/BiasAdd/ReadVariableOpReadVariableOp7sequential_14_conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_14/conv2d_33/BiasAddBiasAdd'sequential_14/conv2d_33/Conv2D:output:06sequential_14/conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������%%@�
-sequential_14/conv2d_33/activation_16/SigmoidSigmoid(sequential_14/conv2d_33/BiasAdd:output:0*
T0*/
_output_shapes
:���������%%@�
)sequential_14/conv2d_33/activation_16/mulMul1sequential_14/conv2d_33/activation_16/Sigmoid:y:0(sequential_14/conv2d_33/BiasAdd:output:0*
T0*/
_output_shapes
:���������%%@�
&sequential_14/max_pooling2d_49/MaxPoolMaxPool-sequential_14/conv2d_33/activation_16/mul:z:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
!sequential_14/dropout_37/IdentityIdentity/sequential_14/max_pooling2d_49/MaxPool:output:0*
T0*/
_output_shapes
:���������@�
-sequential_14/conv2d_34/Conv2D/ReadVariableOpReadVariableOp6sequential_14_conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
sequential_14/conv2d_34/Conv2DConv2D*sequential_14/dropout_37/Identity:output:05sequential_14/conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
.sequential_14/conv2d_34/BiasAdd/ReadVariableOpReadVariableOp7sequential_14_conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_14/conv2d_34/BiasAddBiasAdd'sequential_14/conv2d_34/Conv2D:output:06sequential_14/conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
-sequential_14/conv2d_34/activation_16/SigmoidSigmoid(sequential_14/conv2d_34/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
)sequential_14/conv2d_34/activation_16/mulMul1sequential_14/conv2d_34/activation_16/Sigmoid:y:0(sequential_14/conv2d_34/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
&sequential_14/max_pooling2d_50/MaxPoolMaxPool-sequential_14/conv2d_34/activation_16/mul:z:0*/
_output_shapes
:���������		@*
ksize
*
paddingVALID*
strides
�
!sequential_14/dropout_38/IdentityIdentity/sequential_14/max_pooling2d_50/MaxPool:output:0*
T0*/
_output_shapes
:���������		@�
-sequential_14/conv2d_35/Conv2D/ReadVariableOpReadVariableOp6sequential_14_conv2d_35_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
sequential_14/conv2d_35/Conv2DConv2D*sequential_14/dropout_38/Identity:output:05sequential_14/conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
�
.sequential_14/conv2d_35/BiasAdd/ReadVariableOpReadVariableOp7sequential_14_conv2d_35_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_14/conv2d_35/BiasAddBiasAdd'sequential_14/conv2d_35/Conv2D:output:06sequential_14/conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		��
-sequential_14/conv2d_35/activation_16/SigmoidSigmoid(sequential_14/conv2d_35/BiasAdd:output:0*
T0*0
_output_shapes
:���������		��
)sequential_14/conv2d_35/activation_16/mulMul1sequential_14/conv2d_35/activation_16/Sigmoid:y:0(sequential_14/conv2d_35/BiasAdd:output:0*
T0*0
_output_shapes
:���������		��
&sequential_14/max_pooling2d_51/MaxPoolMaxPool-sequential_14/conv2d_35/activation_16/mul:z:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
�
!sequential_14/dropout_39/IdentityIdentity/sequential_14/max_pooling2d_51/MaxPool:output:0*
T0*0
_output_shapes
:�����������
-sequential_14/conv2d_36/Conv2D/ReadVariableOpReadVariableOp6sequential_14_conv2d_36_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
sequential_14/conv2d_36/Conv2DConv2D*sequential_14/dropout_39/Identity:output:05sequential_14/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
.sequential_14/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp7sequential_14_conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_14/conv2d_36/BiasAddBiasAdd'sequential_14/conv2d_36/Conv2D:output:06sequential_14/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
-sequential_14/conv2d_36/activation_16/SigmoidSigmoid(sequential_14/conv2d_36/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
)sequential_14/conv2d_36/activation_16/mulMul1sequential_14/conv2d_36/activation_16/Sigmoid:y:0(sequential_14/conv2d_36/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
-sequential_14/conv2d_37/Conv2D/ReadVariableOpReadVariableOp6sequential_14_conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
sequential_14/conv2d_37/Conv2DConv2D-sequential_14/conv2d_36/activation_16/mul:z:05sequential_14/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
.sequential_14/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp7sequential_14_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_14/conv2d_37/BiasAddBiasAdd'sequential_14/conv2d_37/Conv2D:output:06sequential_14/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
-sequential_14/conv2d_37/activation_16/SigmoidSigmoid(sequential_14/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
)sequential_14/conv2d_37/activation_16/mulMul1sequential_14/conv2d_37/activation_16/Sigmoid:y:0(sequential_14/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
&sequential_14/max_pooling2d_52/MaxPoolMaxPool-sequential_14/conv2d_37/activation_16/mul:z:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
�
!sequential_14/dropout_40/IdentityIdentity/sequential_14/max_pooling2d_52/MaxPool:output:0*
T0*0
_output_shapes
:����������n
sequential_14/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
sequential_14/flatten_7/ReshapeReshape*sequential_14/dropout_40/Identity:output:0&sequential_14/flatten_7/Const:output:0*
T0*(
_output_shapes
:�����������
xsequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB �
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/ReadVariableOpReadVariableOp�sequential_14_dense_reparameterization_2_independentnormal_constructed_at_dense_reparameterization_2_sample_normal_sample_readvariableop_resource*
_output_shapes
:	�*
dtype0�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOpReadVariableOp�sequential_14_dense_reparameterization_2_independentnormal_constructed_at_dense_reparameterization_2_sample_normal_sample_softplus_readvariableop_resource*
_output_shapes
:	�*
dtype0�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/SoftplusSoftplus�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
}sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/addAddV2�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/add/x:output:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Softplus:activations:0*
T0*
_output_shapes
:	��
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"      �
sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_sliceStridedSlice�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/shape_as_tensor:output:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice/stack:output:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice/stack_1:output:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"      �
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : �
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice_1StridedSlice�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/shape_as_tensor_1:output:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice_1/stack:output:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice_1/stack_1:output:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB �
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/BroadcastArgsBroadcastArgs�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/BroadcastArgs/s0_1:output:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice:output:0*
_output_shapes
:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/BroadcastArgs_1BroadcastArgs�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/BroadcastArgs:r0:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice_1:output:0*
_output_shapes
:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/concatConcatV2�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/concat/values_0:output:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/BroadcastArgs_1:r0:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    �
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/concat:output:0*
T0*#
_output_shapes
:�*
dtype0�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal/mulMul�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal/RandomStandardNormal:output:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal/stddev:output:0*
T0*#
_output_shapes
:��
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normalAddV2�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal/mul:z:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal/mean:output:0*
T0*#
_output_shapes
:��
}sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/mulMul�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal:z:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/add:z:0*
T0*#
_output_shapes
:��
sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/add_1AddV2�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/mul:z:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/ReadVariableOp:value:0*
T0*#
_output_shapes
:��
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/ReshapeReshape�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/add_1:z:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Reshape/shape:output:0*
T0*#
_output_shapes
:��
ysequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
ssequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/ReshapeReshape�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Reshape:output:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Reshape/shape:output:0*
T0*
_output_shapes
:	��
/sequential_14/dense_reparameterization_2/MatMulMatMul(sequential_14/flatten_7/Reshape:output:0|sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Reshape:output:0*
T0*'
_output_shapes
:����������
zsequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB �
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/ReadVariableOpReadVariableOp�sequential_14_dense_reparameterization_2_independentnormal_constructed_at_dense_reparameterization_2_sample_1_normal_sample_readvariableop_resource*
_output_shapes
:*
dtype0�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOpReadVariableOp�sequential_14_dense_reparameterization_2_independentnormal_constructed_at_dense_reparameterization_2_sample_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
:*
dtype0�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/SoftplusSoftplus�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/addAddV2�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/add/x:output:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Softplus:activations:0*
T0*
_output_shapes
:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_sliceStridedSlice�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/shape_as_tensor:output:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice/stack:output:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice/stack_1:output:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : �
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice_1StridedSlice�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/shape_as_tensor_1:output:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack:output:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack_1:output:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB �
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/BroadcastArgsBroadcastArgs�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/BroadcastArgs/s0_1:output:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice:output:0*
_output_shapes
:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/BroadcastArgs_1BroadcastArgs�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/BroadcastArgs:r0:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/concatConcatV2�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/concat/values_0:output:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/BroadcastArgs_1:r0:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    �
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/concat:output:0*
T0*
_output_shapes

:*
dtype0�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal/mulMul�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal/stddev:output:0*
T0*
_output_shapes

:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normalAddV2�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal/mul:z:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal/mean:output:0*
T0*
_output_shapes

:�
sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/mulMul�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal:z:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/add:z:0*
T0*
_output_shapes

:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/add_1AddV2�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/mul:z:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/ReadVariableOp:value:0*
T0*
_output_shapes

:�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/ReshapeReshape�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/add_1:z:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

:�
{sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
usequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/ReshapeReshape�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Reshape:output:0�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Reshape/shape:output:0*
T0*
_output_shapes
:�
0sequential_14/dense_reparameterization_2/BiasAddBiasAdd9sequential_14/dense_reparameterization_2/MatMul:product:0~sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Reshape:output:0*
T0*'
_output_shapes
:����������
�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp�sequential_14_dense_reparameterization_2_independentnormal_constructed_at_dense_reparameterization_2_sample_normal_sample_softplus_readvariableop_resource*
_output_shapes
:	�*
dtype0�
�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes
:	��
�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes
:	��
�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log�sequential_14_dense_reparameterization_2_kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2147682*
T0*
_output_shapes
: �
�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes
:	��
�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp�sequential_14_dense_reparameterization_2_independentnormal_constructed_at_dense_reparameterization_2_sample_normal_sample_readvariableop_resource*
_output_shapes
:	�*
dtype0�
�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0�sequential_14_dense_reparameterization_2_kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2147682*
T0*
_output_shapes
:	��
�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv�sequential_14_dense_reparameterization_2_kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�sequential_14_dense_reparameterization_2_kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2147682*
T0*
_output_shapes
:	��
�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes
:	��
�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes
:	��
�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:	��
�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes
:	��
�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes
:	��
�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes
:	��
�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:	��
�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"���������
�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: w
2sequential_14/dense_reparameterization_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * H[F�
0sequential_14/dense_reparameterization_2/truedivRealDiv�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0;sequential_14/dense_reparameterization_2/truediv/y:output:0*
T0*
_output_shapes
: �
:sequential_14/dense_reparameterization_2/divergence_kernelIdentity4sequential_14/dense_reparameterization_2/truediv:z:0*
T0*
_output_shapes
: �
�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp�sequential_14_dense_reparameterization_2_independentnormal_constructed_at_dense_reparameterization_2_sample_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
:*
dtype0�
�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:�
�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes
:�
�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes
:�
�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log�sequential_14_dense_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2147710*
T0*
_output_shapes
: �
�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes
:�
�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp�sequential_14_dense_reparameterization_2_independentnormal_constructed_at_dense_reparameterization_2_sample_1_normal_sample_readvariableop_resource*
_output_shapes
:*
dtype0�
�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0�sequential_14_dense_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2147710*
T0*
_output_shapes
:�
�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv�sequential_14_dense_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�sequential_14_dense_reparameterization_2_kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2147710*
T0*
_output_shapes
:�
�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes
:�
�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes
:�
�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:�
�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes
:�
�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes
:�
�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes
:�
�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:�
�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: y
4sequential_14/dense_reparameterization_2/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 * H[F�
2sequential_14/dense_reparameterization_2/truediv_1RealDiv�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0=sequential_14/dense_reparameterization_2/truediv_1/y:output:0*
T0*
_output_shapes
: �
8sequential_14/dense_reparameterization_2/divergence_biasIdentity6sequential_14/dense_reparameterization_2/truediv_1:z:0*
T0*
_output_shapes
: �
Zsequential_14/one_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/IdentityIdentity9sequential_14/dense_reparameterization_2/BiasAdd:output:0*
T0*'
_output_shapes
:����������
bsequential_14/one_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
����������
Xsequential_14/one_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/ArgMaxArgMaxcsequential_14/one_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/Identity:output:0ksequential_14/one_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:����������
bsequential_14/one_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
csequential_14/one_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
_sequential_14/one_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :�
Ysequential_14/one_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/one_hotOneHotasequential_14/one_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/ArgMax:output:0hsequential_14/one_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/one_hot/depth:output:0ksequential_14/one_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/one_hot/on_value:output:0lsequential_14/one_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/one_hot/off_value:output:0*
T0*'
_output_shapes
:����������
IdentityIdentitybsequential_14/one_hot_categorical_7/tensor_coercible/value/OneHotCategorical/mode/one_hot:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^sequential_14/conv2d_33/BiasAdd/ReadVariableOp.^sequential_14/conv2d_33/Conv2D/ReadVariableOp/^sequential_14/conv2d_34/BiasAdd/ReadVariableOp.^sequential_14/conv2d_34/Conv2D/ReadVariableOp/^sequential_14/conv2d_35/BiasAdd/ReadVariableOp.^sequential_14/conv2d_35/Conv2D/ReadVariableOp/^sequential_14/conv2d_36/BiasAdd/ReadVariableOp.^sequential_14/conv2d_36/Conv2D/ReadVariableOp/^sequential_14/conv2d_37/BiasAdd/ReadVariableOp.^sequential_14/conv2d_37/Conv2D/ReadVariableOp�^sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/ReadVariableOp�^sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOp�^sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/ReadVariableOp�^sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp�^sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�^sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�^sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�^sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�^sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/ReadVariableOp�^sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOp�^sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/ReadVariableOp�^sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp�^sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�^sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�^sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�^sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp9^sequential_14/depthwise_conv2d_13/BiasAdd/ReadVariableOp;^sequential_14/depthwise_conv2d_13/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:�����������: : : : : : : : : : : : : : : : : : : : : : : : : :	�: :2`
.sequential_14/conv2d_33/BiasAdd/ReadVariableOp.sequential_14/conv2d_33/BiasAdd/ReadVariableOp2^
-sequential_14/conv2d_33/Conv2D/ReadVariableOp-sequential_14/conv2d_33/Conv2D/ReadVariableOp2`
.sequential_14/conv2d_34/BiasAdd/ReadVariableOp.sequential_14/conv2d_34/BiasAdd/ReadVariableOp2^
-sequential_14/conv2d_34/Conv2D/ReadVariableOp-sequential_14/conv2d_34/Conv2D/ReadVariableOp2`
.sequential_14/conv2d_35/BiasAdd/ReadVariableOp.sequential_14/conv2d_35/BiasAdd/ReadVariableOp2^
-sequential_14/conv2d_35/Conv2D/ReadVariableOp-sequential_14/conv2d_35/Conv2D/ReadVariableOp2`
.sequential_14/conv2d_36/BiasAdd/ReadVariableOp.sequential_14/conv2d_36/BiasAdd/ReadVariableOp2^
-sequential_14/conv2d_36/Conv2D/ReadVariableOp-sequential_14/conv2d_36/Conv2D/ReadVariableOp2`
.sequential_14/conv2d_37/BiasAdd/ReadVariableOp.sequential_14/conv2d_37/BiasAdd/ReadVariableOp2^
-sequential_14/conv2d_37/Conv2D/ReadVariableOp-sequential_14/conv2d_37/Conv2D/ReadVariableOp2�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/ReadVariableOp�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/ReadVariableOp2�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOp�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOp2�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/ReadVariableOp�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/ReadVariableOp2�
�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp�sequential_14/conv2d_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp2�
�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2�
�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�sequential_14/conv2d_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2�
�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2�
�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�sequential_14/conv2d_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/ReadVariableOp�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/ReadVariableOp2�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOp�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOp2�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/ReadVariableOp�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/ReadVariableOp2�
�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp�sequential_14/dense_reparameterization_2/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp2�
�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2�
�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�sequential_14/dense_reparameterization_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2�
�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2�
�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�sequential_14/dense_reparameterization_2/KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2t
8sequential_14/depthwise_conv2d_13/BiasAdd/ReadVariableOp8sequential_14/depthwise_conv2d_13/BiasAdd/ReadVariableOp2x
:sequential_14/depthwise_conv2d_13/depthwise/ReadVariableOp:sequential_14/depthwise_conv2d_13/depthwise/ReadVariableOp: 

_output_shapes
::

_output_shapes
: :%!

_output_shapes
:	�:

_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :t p
1
_output_shapes
:�����������
;
_user_specified_name#!conv2d_reparameterization_2_input
�
i
M__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_2150268

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

f
G__inference_dropout_40_layer_call_and_return_conditional_losses_2148171

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentitydropout/SelectV2:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_37_layer_call_and_return_conditional_losses_2150511

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
activation_16/SigmoidSigmoidBiasAdd:output:0*
T0*0
_output_shapes
:�����������
activation_16/mulMulactivation_16/Sigmoid:y:0BiasAdd:output:0*
T0*0
_output_shapes
:����������m
IdentityIdentityactivation_16/mul:z:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
G__inference_dropout_39_layer_call_and_return_conditional_losses_2150469

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:����������d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_36_layer_call_and_return_conditional_losses_2148134

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
activation_16/SigmoidSigmoidBiasAdd:output:0*
T0*0
_output_shapes
:�����������
activation_16/mulMulactivation_16/Sigmoid:y:0BiasAdd:output:0*
T0*0
_output_shapes
:����������m
IdentityIdentityactivation_16/mul:z:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

f
G__inference_dropout_39_layer_call_and_return_conditional_losses_2150464

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentitydropout/SelectV2:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
i
M__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_2150384

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
e
G__inference_dropout_37_layer_call_and_return_conditional_losses_2148410

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
F__inference_conv2d_36_layer_call_and_return_conditional_losses_2150490

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
activation_16/SigmoidSigmoidBiasAdd:output:0*
T0*0
_output_shapes
:�����������
activation_16/mulMulactivation_16/Sigmoid:y:0BiasAdd:output:0*
T0*0
_output_shapes
:����������m
IdentityIdentityactivation_16/mul:z:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

f
G__inference_dropout_39_layer_call_and_return_conditional_losses_2148120

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentitydropout/SelectV2:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
7__inference_one_hot_categorical_7_layer_call_fn_2150733

inputs
identity

identity_1�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_one_hot_categorical_7_layer_call_and_return_conditional_losses_2148357`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:���������"!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
H
,__inference_dropout_39_layer_call_fn_2150452

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_39_layer_call_and_return_conditional_losses_2148434i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
<__inference_dense_reparameterization_2_layer_call_fn_2150582

inputs
unknown:	�
	unknown_0:	�
	unknown_1:
	unknown_2:
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������: : *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *`
f[RY
W__inference_dense_reparameterization_2_layer_call_and_return_conditional_losses_2148325o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:����������: : : : : :	�: :22
StatefulPartitionedCallStatefulPartitionedCall: 

_output_shapes
::

_output_shapes
: :%!

_output_shapes
:	�:

_output_shapes
: :P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
G__inference_dropout_36_layer_call_and_return_conditional_losses_2150295

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:���������%%�d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:���������%%�"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������%%�:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
�
�
F__inference_conv2d_33_layer_call_and_return_conditional_losses_2150316

inputs9
conv2d_readvariableop_resource:�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������%%@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������%%@l
activation_16/SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:���������%%@
activation_16/mulMulactivation_16/Sigmoid:y:0BiasAdd:output:0*
T0*/
_output_shapes
:���������%%@l
IdentityIdentityactivation_16/mul:z:0^NoOp*
T0*/
_output_shapes
:���������%%@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������%%�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
��
�
X__inference_conv2d_reparameterization_2_layer_call_and_return_conditional_losses_2150225

inputs�
iindependentnormal_constructed_at_conv2d_reparameterization_2_sample_normal_sample_readvariableop_resource: �
rindependentnormal_constructed_at_conv2d_reparameterization_2_sample_normal_sample_softplus_readvariableop_resource: y
kindependentnormal_constructed_at_conv2d_reparameterization_2_sample_1_normal_sample_readvariableop_resource: �
tindependentnormal_constructed_at_conv2d_reparameterization_2_sample_1_normal_sample_softplus_readvariableop_resource: �
�kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2150166�
�kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�
�kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2150194�
�kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity

identity_1

identity_2��`IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/ReadVariableOp�iIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOp�bIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/ReadVariableOp�kIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp��KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp��KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp��KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp��KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�
PIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB �
^IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :�
oIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
`IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/ReadVariableOpReadVariableOpiindependentnormal_constructed_at_conv2d_reparameterization_2_sample_normal_sample_readvariableop_resource*&
_output_shapes
: *
dtype0�
iIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOpReadVariableOprindependentnormal_constructed_at_conv2d_reparameterization_2_sample_normal_sample_softplus_readvariableop_resource*&
_output_shapes
: *
dtype0�
ZIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/SoftplusSoftplusqIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
WIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
UIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/addAddV2`IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/add/x:output:0hIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Softplus:activations:0*
T0*&
_output_shapes
: �
aIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"             �
WIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
eIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
gIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
gIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
_IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_sliceStridedSlicejIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/shape_as_tensor:output:0nIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice/stack:output:0pIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice/stack_1:output:0pIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
cIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
YIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : �
gIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
iIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
iIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
aIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice_1StridedSlicelIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/shape_as_tensor_1:output:0pIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice_1/stack:output:0rIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice_1/stack_1:output:0rIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
bIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB �
dIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
_IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/BroadcastArgsBroadcastArgsmIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/BroadcastArgs/s0_1:output:0hIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice:output:0*
_output_shapes
:�
aIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/BroadcastArgs_1BroadcastArgsdIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/BroadcastArgs:r0:0jIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/strided_slice_1:output:0*
_output_shapes
:�
aIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:�
]IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
XIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/concatConcatV2jIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/concat/values_0:output:0fIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/BroadcastArgs_1:r0:0fIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
kIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    �
mIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
{IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalaIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/concat:output:0*
T0**
_output_shapes
: *
dtype0�
jIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal/mulMul�IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal/RandomStandardNormal:output:0vIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal/stddev:output:0*
T0**
_output_shapes
: �
fIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normalAddV2nIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal/mul:z:0tIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal/mean:output:0*
T0**
_output_shapes
: �
UIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/mulMuljIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/normal/random_normal:z:0YIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/add:z:0*
T0**
_output_shapes
: �
WIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/add_1AddV2YIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/mul:z:0hIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/ReadVariableOp:value:0*
T0**
_output_shapes
: �
_IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"                �
YIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/ReshapeReshape[IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/add_1:z:0hIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Reshape/shape:output:0*
T0**
_output_shapes
: �
QIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
KIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/ReshapeReshapebIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Reshape:output:0ZIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Reshape/shape:output:0*
T0*&
_output_shapes
: �
Conv2DConv2DinputsTIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Reshape:output:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
�
RIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB �
`IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :�
qIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
bIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/ReadVariableOpReadVariableOpkindependentnormal_constructed_at_conv2d_reparameterization_2_sample_1_normal_sample_readvariableop_resource*
_output_shapes
: *
dtype0�
kIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOpReadVariableOptindependentnormal_constructed_at_conv2d_reparameterization_2_sample_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
: *
dtype0�
\IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/SoftplusSoftplussIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
: �
YIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
WIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/addAddV2bIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/add/x:output:0jIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Softplus:activations:0*
T0*
_output_shapes
: �
cIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: �
YIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
gIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
iIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
iIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
aIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_sliceStridedSlicelIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/shape_as_tensor:output:0pIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice/stack:output:0rIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice/stack_1:output:0rIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
eIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB: �
[IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : �
iIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
kIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
kIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
cIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice_1StridedSlicenIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/shape_as_tensor_1:output:0rIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack:output:0tIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack_1:output:0tIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
dIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB �
fIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
aIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/BroadcastArgsBroadcastArgsoIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/BroadcastArgs/s0_1:output:0jIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice:output:0*
_output_shapes
:�
cIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/BroadcastArgs_1BroadcastArgsfIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/BroadcastArgs:r0:0lIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:�
cIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:�
_IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
ZIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/concatConcatV2lIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/concat/values_0:output:0hIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/BroadcastArgs_1:r0:0hIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
mIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    �
oIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
}IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalcIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/concat:output:0*
T0*
_output_shapes

: *
dtype0�
lIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal/mulMul�IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0xIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal/stddev:output:0*
T0*
_output_shapes

: �
hIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normalAddV2pIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal/mul:z:0vIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal/mean:output:0*
T0*
_output_shapes

: �
WIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/mulMullIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/normal/random_normal:z:0[IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/add:z:0*
T0*
_output_shapes

: �
YIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/add_1AddV2[IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/mul:z:0jIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/ReadVariableOp:value:0*
T0*
_output_shapes

: �
aIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       �
[IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/ReshapeReshape]IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/add_1:z:0jIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

: �
SIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB: �
MIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/ReshapeReshapedIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Reshape:output:0\IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Reshape/shape:output:0*
T0*
_output_shapes
: �
BiasAddBiasAddConv2D:output:0VIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Reshape:output:0*
T0*1
_output_shapes
:����������� Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:����������� �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOprindependentnormal_constructed_at_conv2d_reparameterization_2_sample_normal_sample_softplus_readvariableop_resource*&
_output_shapes
: *
dtype0�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log�kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2150166*
T0*
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOpiindependentnormal_constructed_at_conv2d_reparameterization_2_sample_normal_sample_readvariableop_resource*&
_output_shapes
: *
dtype0�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0�kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2150166*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv�kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�kullbackleibler_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2150166*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"�����������������
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * H[F�
truedivRealDiv�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0truediv/y:output:0*
T0*
_output_shapes
: K
divergence_kernelIdentitytruediv:z:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOptindependentnormal_constructed_at_conv2d_reparameterization_2_sample_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
: *
dtype0�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log�kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2150194*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOpkindependentnormal_constructed_at_conv2d_reparameterization_2_sample_1_normal_sample_readvariableop_resource*
_output_shapes
: *
dtype0�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0�kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2150194*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv�kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�kullbackleibler_1_independentnormal_constructed_at_conv2d_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2150194*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 * H[F�
	truediv_1RealDiv�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0truediv_1/y:output:0*
T0*
_output_shapes
: K
divergence_biasIdentitytruediv_1:z:0*
T0*
_output_shapes
: k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:����������� Z

Identity_1Identitydivergence_kernel:output:0^NoOp*
T0*
_output_shapes
: X

Identity_2Identitydivergence_bias:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOpa^IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/ReadVariableOpj^IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOpc^IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/ReadVariableOpl^IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp�^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�^KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�^KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:�����������: : : : : : : : 2�
`IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/ReadVariableOp`IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/ReadVariableOp2�
iIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOpiIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOp2�
bIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/ReadVariableOpbIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/ReadVariableOp2�
kIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOpkIndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp2�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_conv2d_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

f
G__inference_dropout_36_layer_call_and_return_conditional_losses_2150290

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:���������%%�Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:���������%%�*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:���������%%�T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*0
_output_shapes
:���������%%�j
IdentityIdentitydropout/SelectV2:output:0*
T0*0
_output_shapes
:���������%%�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������%%�:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
�
i
M__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_2147776

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
e
G__inference_dropout_38_layer_call_and_return_conditional_losses_2148422

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������		@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������		@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������		@:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
H
,__inference_dropout_37_layer_call_fn_2150336

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_37_layer_call_and_return_conditional_losses_2148410h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�

f
G__inference_dropout_36_layer_call_and_return_conditional_losses_2148021

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:���������%%�Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:���������%%�*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:���������%%�T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*0
_output_shapes
:���������%%�j
IdentityIdentitydropout/SelectV2:output:0*
T0*0
_output_shapes
:���������%%�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������%%�:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
��
�
W__inference_dense_reparameterization_2_layer_call_and_return_conditional_losses_2150726

inputs{
hindependentnormal_constructed_at_dense_reparameterization_2_sample_normal_sample_readvariableop_resource:	��
qindependentnormal_constructed_at_dense_reparameterization_2_sample_normal_sample_softplus_readvariableop_resource:	�x
jindependentnormal_constructed_at_dense_reparameterization_2_sample_1_normal_sample_readvariableop_resource:�
sindependentnormal_constructed_at_dense_reparameterization_2_sample_1_normal_sample_softplus_readvariableop_resource:�
�kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2150667�
�kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�
�kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2150695�
�kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity

identity_1

identity_2��_IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/ReadVariableOp�hIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOp�aIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/ReadVariableOp�jIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp��KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp��KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp��KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp��KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�
OIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB �
]IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :�
nIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
_IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/ReadVariableOpReadVariableOphindependentnormal_constructed_at_dense_reparameterization_2_sample_normal_sample_readvariableop_resource*
_output_shapes
:	�*
dtype0�
hIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOpReadVariableOpqindependentnormal_constructed_at_dense_reparameterization_2_sample_normal_sample_softplus_readvariableop_resource*
_output_shapes
:	�*
dtype0�
YIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/SoftplusSoftpluspIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
VIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
TIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/addAddV2_IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/add/x:output:0gIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Softplus:activations:0*
T0*
_output_shapes
:	��
`IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"      �
VIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
dIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
fIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
fIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
^IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_sliceStridedSliceiIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/shape_as_tensor:output:0mIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice/stack:output:0oIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice/stack_1:output:0oIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
bIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"      �
XIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : �
fIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
hIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
hIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
`IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice_1StridedSlicekIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/shape_as_tensor_1:output:0oIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice_1/stack:output:0qIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice_1/stack_1:output:0qIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
aIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB �
cIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
^IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/BroadcastArgsBroadcastArgslIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/BroadcastArgs/s0_1:output:0gIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice:output:0*
_output_shapes
:�
`IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/BroadcastArgs_1BroadcastArgscIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/BroadcastArgs:r0:0iIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/strided_slice_1:output:0*
_output_shapes
:�
`IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:�
\IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
WIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/concatConcatV2iIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/concat/values_0:output:0eIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/BroadcastArgs_1:r0:0eIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
jIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
zIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal`IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/concat:output:0*
T0*#
_output_shapes
:�*
dtype0�
iIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal/mulMul�IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal/RandomStandardNormal:output:0uIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal/stddev:output:0*
T0*#
_output_shapes
:��
eIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normalAddV2mIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal/mul:z:0sIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal/mean:output:0*
T0*#
_output_shapes
:��
TIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/mulMuliIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/normal/random_normal:z:0XIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/add:z:0*
T0*#
_output_shapes
:��
VIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/add_1AddV2XIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/mul:z:0gIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/ReadVariableOp:value:0*
T0*#
_output_shapes
:��
^IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
XIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/ReshapeReshapeZIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/add_1:z:0gIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Reshape/shape:output:0*
T0*#
_output_shapes
:��
PIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
JIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/ReshapeReshapeaIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Reshape:output:0YIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Reshape/shape:output:0*
T0*
_output_shapes
:	��
MatMulMatMulinputsSIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Reshape:output:0*
T0*'
_output_shapes
:����������
QIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB �
_IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :�
pIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
aIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/ReadVariableOpReadVariableOpjindependentnormal_constructed_at_dense_reparameterization_2_sample_1_normal_sample_readvariableop_resource*
_output_shapes
:*
dtype0�
jIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOpReadVariableOpsindependentnormal_constructed_at_dense_reparameterization_2_sample_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
:*
dtype0�
[IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/SoftplusSoftplusrIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:�
XIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
VIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/addAddV2aIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/add/x:output:0iIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Softplus:activations:0*
T0*
_output_shapes
:�
bIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:�
XIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
fIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
hIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
hIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
`IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_sliceStridedSlicekIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/shape_as_tensor:output:0oIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice/stack:output:0qIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice/stack_1:output:0qIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
dIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:�
ZIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : �
hIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
jIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
jIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
bIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice_1StridedSlicemIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/shape_as_tensor_1:output:0qIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack:output:0sIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack_1:output:0sIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
cIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB �
eIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
`IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/BroadcastArgsBroadcastArgsnIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/BroadcastArgs/s0_1:output:0iIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice:output:0*
_output_shapes
:�
bIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/BroadcastArgs_1BroadcastArgseIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/BroadcastArgs:r0:0kIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:�
bIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:�
^IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
YIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/concatConcatV2kIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/concat/values_0:output:0gIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/BroadcastArgs_1:r0:0gIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
lIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    �
nIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
|IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalbIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/concat:output:0*
T0*
_output_shapes

:*
dtype0�
kIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal/mulMul�IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0wIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal/stddev:output:0*
T0*
_output_shapes

:�
gIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normalAddV2oIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal/mul:z:0uIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal/mean:output:0*
T0*
_output_shapes

:�
VIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/mulMulkIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/normal/random_normal:z:0ZIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/add:z:0*
T0*
_output_shapes

:�
XIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/add_1AddV2ZIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/mul:z:0iIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/ReadVariableOp:value:0*
T0*
_output_shapes

:�
`IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
ZIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/ReshapeReshape\IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/add_1:z:0iIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

:�
RIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
LIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/ReshapeReshapecIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Reshape:output:0[IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Reshape/shape:output:0*
T0*
_output_shapes
:�
BiasAddBiasAddMatMul:product:0UIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Reshape:output:0*
T0*'
_output_shapes
:����������
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOpqindependentnormal_constructed_at_dense_reparameterization_2_sample_normal_sample_softplus_readvariableop_resource*
_output_shapes
:	�*
dtype0�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log�kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2150667*
T0*
_output_shapes
: �
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOphindependentnormal_constructed_at_dense_reparameterization_2_sample_normal_sample_readvariableop_resource*
_output_shapes
:	�*
dtype0�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0�kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2150667*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv�kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�kullbackleibler_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2150667*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:	��
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"���������
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * H[F�
truedivRealDiv�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0truediv/y:output:0*
T0*
_output_shapes
: K
divergence_kernelIdentitytruediv:z:0*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOpsindependentnormal_constructed_at_dense_reparameterization_2_sample_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
:*
dtype0�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log�kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2150695*
T0*
_output_shapes
: �
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOpjindependentnormal_constructed_at_dense_reparameterization_2_sample_1_normal_sample_readvariableop_resource*
_output_shapes
:*
dtype0�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0�kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2150695*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv�kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x�kullbackleibler_1_independentnormal_constructed_at_dense_reparameterization_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2150695*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 * H[F�
	truediv_1RealDiv�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0truediv_1/y:output:0*
T0*
_output_shapes
: K
divergence_biasIdentitytruediv_1:z:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identitydivergence_kernel:output:0^NoOp*
T0*
_output_shapes
: X

Identity_2Identitydivergence_bias:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp`^IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/ReadVariableOpi^IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOpb^IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/ReadVariableOpk^IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp�^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�^KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�^KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:����������: : : : : :	�: :2�
_IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/ReadVariableOp_IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/ReadVariableOp2�
hIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOphIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample/Normal/sample/Softplus/ReadVariableOp2�
aIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/ReadVariableOpaIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/ReadVariableOp2�
jIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOpjIndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/sample_1/Normal/sample/Softplus/ReadVariableOp2�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2�
�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2�
�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp�KullbackLeibler_1/IndependentNormal_CONSTRUCTED_AT_dense_reparameterization_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp: 

_output_shapes
::

_output_shapes
: :%!

_output_shapes
:	�:

_output_shapes
: :P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
i
M__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_2147800

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
e
,__inference_dropout_38_layer_call_fn_2150389

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������		@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_38_layer_call_and_return_conditional_losses_2148087w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������		@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������		@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
H
,__inference_dropout_40_layer_call_fn_2150531

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_40_layer_call_and_return_conditional_losses_2148451i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_35_layer_call_and_return_conditional_losses_2150432

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�m
activation_16/SigmoidSigmoidBiasAdd:output:0*
T0*0
_output_shapes
:���������		��
activation_16/mulMulactivation_16/Sigmoid:y:0BiasAdd:output:0*
T0*0
_output_shapes
:���������		�m
IdentityIdentityactivation_16/mul:z:0^NoOp*
T0*0
_output_shapes
:���������		�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������		@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������		@
 
_user_specified_nameinputs
�
�
F__inference_conv2d_33_layer_call_and_return_conditional_losses_2148035

inputs9
conv2d_readvariableop_resource:�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������%%@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������%%@l
activation_16/SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:���������%%@
activation_16/mulMulactivation_16/Sigmoid:y:0BiasAdd:output:0*
T0*/
_output_shapes
:���������%%@l
IdentityIdentityactivation_16/mul:z:0^NoOp*
T0*/
_output_shapes
:���������%%@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������%%�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������%%�
 
_user_specified_nameinputs
�
�
P__inference_depthwise_conv2d_13_layer_call_and_return_conditional_losses_2150258

inputs;
!depthwise_readvariableop_resource: .
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�depthwise/ReadVariableOp�
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
: *
dtype0h
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             h
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAdddepthwise:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������KK�m
activation_16/SigmoidSigmoidBiasAdd:output:0*
T0*0
_output_shapes
:���������KK��
activation_16/mulMulactivation_16/Sigmoid:y:0BiasAdd:output:0*
T0*0
_output_shapes
:���������KK�m
IdentityIdentityactivation_16/mul:z:0^NoOp*
T0*0
_output_shapes
:���������KK�z
NoOpNoOp^BiasAdd/ReadVariableOp^depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������KK : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:W S
/
_output_shapes
:���������KK 
 
_user_specified_nameinputs
�
i
M__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_2147752

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
y
!conv2d_reparameterization_2_inputT
3serving_default_conv2d_reparameterization_2_input:0�����������I
one_hot_categorical_70
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer-12
layer-13
layer_with_weights-5
layer-14
layer_with_weights-6
layer-15
layer-16
layer-17
layer-18
layer_with_weights-7
layer-19
layer-20
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%kernel_posterior_loc
(&$kernel_posterior_untransformed_scale
'kernel_posterior
(kernel_prior
)bias_posterior_loc
&*"bias_posterior_untransformed_scale
+bias_posterior
,
bias_prior"
_tf_keras_layer
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
9
activation
:depthwise_kernel
;bias
 <_jit_compiled_convolution_op"
_tf_keras_layer
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
I_random_generator"
_tf_keras_layer
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
9
activation

Pkernel
Qbias
 R_jit_compiled_convolution_op"
_tf_keras_layer
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses
__random_generator"
_tf_keras_layer
�
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses
9
activation

fkernel
gbias
 h_jit_compiled_convolution_op"
_tf_keras_layer
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
�
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses
u_random_generator"
_tf_keras_layer
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses
9
activation

|kernel
}bias
 ~_jit_compiled_convolution_op"
_tf_keras_layer
�
	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
9
activation
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
9
activation
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel_posterior_loc
)�$kernel_posterior_untransformed_scale
�kernel_posterior
�kernel_prior
�bias_posterior_loc
'�"bias_posterior_untransformed_scale
�bias_posterior
�
bias_prior"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_kwargs"
_tf_keras_layer
�
%0
&1
)2
*3
:4
;5
P6
Q7
f8
g9
|10
}11
�12
�13
�14
�15
�16
�17
�18
�19"
trackable_list_wrapper
�
%0
&1
)2
*3
:4
;5
P6
Q7
f8
g9
|10
}11
�12
�13
�14
�15
�16
�17
�18
�19"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
/__inference_sequential_14_layer_call_fn_2148648
/__inference_sequential_14_layer_call_fn_2148803
/__inference_sequential_14_layer_call_fn_2149247
/__inference_sequential_14_layer_call_fn_2149312�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
J__inference_sequential_14_layer_call_and_return_conditional_losses_2148365
J__inference_sequential_14_layer_call_and_return_conditional_losses_2148492
J__inference_sequential_14_layer_call_and_return_conditional_losses_2149702
J__inference_sequential_14_layer_call_and_return_conditional_losses_2150057�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�
capture_24
�
capture_25
�
capture_26
�
capture_27B�
"__inference__wrapped_model_2147746!conv2d_reparameterization_2_input"�
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
 z�	capture_4z�	capture_5z�	capture_6z�	capture_7z�
capture_24z�
capture_25z�
capture_26z�
capture_27
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
signature_map
<
%0
&1
)2
*3"
trackable_list_wrapper
<
%0
&1
)2
*3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
=__inference_conv2d_reparameterization_2_layer_call_fn_2150080�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
X__inference_conv2d_reparameterization_2_layer_call_and_return_conditional_losses_2150225�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
J:H 20conv2d_reparameterization_2/kernel_posterior_loc
Z:X 2@conv2d_reparameterization_2/kernel_posterior_untransformed_scale
G
�_distribution
�_graph_parents"
_generic_user_object
G
�_distribution
�_graph_parents"
_generic_user_object
<:: 2.conv2d_reparameterization_2/bias_posterior_loc
L:J 2>conv2d_reparameterization_2/bias_posterior_untransformed_scale
G
�_distribution
�_graph_parents"
_generic_user_object
G
�_distribution
�_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
2__inference_max_pooling2d_47_layer_call_fn_2150230�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
M__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_2150235�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
5__inference_depthwise_conv2d_13_layer_call_fn_2150244�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
P__inference_depthwise_conv2d_13_layer_call_and_return_conditional_losses_2150258�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
>:< 2$depthwise_conv2d_13/depthwise_kernel
':%�2depthwise_conv2d_13/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
2__inference_max_pooling2d_48_layer_call_fn_2150263�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
M__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_2150268�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_dropout_36_layer_call_fn_2150273
,__inference_dropout_36_layer_call_fn_2150278�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_dropout_36_layer_call_and_return_conditional_losses_2150290
G__inference_dropout_36_layer_call_and_return_conditional_losses_2150295�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_conv2d_33_layer_call_fn_2150304�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
F__inference_conv2d_33_layer_call_and_return_conditional_losses_2150316�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
+:)�@2conv2d_33/kernel
:@2conv2d_33/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
2__inference_max_pooling2d_49_layer_call_fn_2150321�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
M__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_2150326�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_dropout_37_layer_call_fn_2150331
,__inference_dropout_37_layer_call_fn_2150336�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_dropout_37_layer_call_and_return_conditional_losses_2150348
G__inference_dropout_37_layer_call_and_return_conditional_losses_2150353�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_conv2d_34_layer_call_fn_2150362�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
F__inference_conv2d_34_layer_call_and_return_conditional_losses_2150374�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
*:(@@2conv2d_34/kernel
:@2conv2d_34/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
2__inference_max_pooling2d_50_layer_call_fn_2150379�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
M__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_2150384�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_dropout_38_layer_call_fn_2150389
,__inference_dropout_38_layer_call_fn_2150394�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_dropout_38_layer_call_and_return_conditional_losses_2150406
G__inference_dropout_38_layer_call_and_return_conditional_losses_2150411�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
|0
}1"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_conv2d_35_layer_call_fn_2150420�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
F__inference_conv2d_35_layer_call_and_return_conditional_losses_2150432�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
+:)@�2conv2d_35/kernel
:�2conv2d_35/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
2__inference_max_pooling2d_51_layer_call_fn_2150437�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
M__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_2150442�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_dropout_39_layer_call_fn_2150447
,__inference_dropout_39_layer_call_fn_2150452�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_dropout_39_layer_call_and_return_conditional_losses_2150464
G__inference_dropout_39_layer_call_and_return_conditional_losses_2150469�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_conv2d_36_layer_call_fn_2150478�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
F__inference_conv2d_36_layer_call_and_return_conditional_losses_2150490�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
,:*��2conv2d_36/kernel
:�2conv2d_36/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_conv2d_37_layer_call_fn_2150499�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
F__inference_conv2d_37_layer_call_and_return_conditional_losses_2150511�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
,:*��2conv2d_37/kernel
:�2conv2d_37/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
2__inference_max_pooling2d_52_layer_call_fn_2150516�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
M__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_2150521�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_dropout_40_layer_call_fn_2150526
,__inference_dropout_40_layer_call_fn_2150531�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_dropout_40_layer_call_and_return_conditional_losses_2150543
G__inference_dropout_40_layer_call_and_return_conditional_losses_2150548�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_flatten_7_layer_call_fn_2150553�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
F__inference_flatten_7_layer_call_and_return_conditional_losses_2150559�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
@
�0
�1
�2
�3"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
<__inference_dense_reparameterization_2_layer_call_fn_2150582�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
W__inference_dense_reparameterization_2_layer_call_and_return_conditional_losses_2150726�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
B:@	�2/dense_reparameterization_2/kernel_posterior_loc
R:P	�2?dense_reparameterization_2/kernel_posterior_untransformed_scale
G
�_distribution
�_graph_parents"
_generic_user_object
G
�_distribution
�_graph_parents"
_generic_user_object
;:92-dense_reparameterization_2/bias_posterior_loc
K:I2=dense_reparameterization_2/bias_posterior_untransformed_scale
G
�_distribution
�_graph_parents"
_generic_user_object
G
�_distribution
�_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_one_hot_categorical_7_layer_call_fn_2150733
7__inference_one_hot_categorical_7_layer_call_fn_2150740�
���
FullArgSpec
args�

jinputs
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_one_hot_categorical_7_layer_call_and_return_conditional_losses_2150751
R__inference_one_hot_categorical_7_layer_call_and_return_conditional_losses_2150762�
���
FullArgSpec
args�

jinputs
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�
capture_24
�
capture_25
�
capture_26
�
capture_27B�
/__inference_sequential_14_layer_call_fn_2148648!conv2d_reparameterization_2_input"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_4z�	capture_5z�	capture_6z�	capture_7z�
capture_24z�
capture_25z�
capture_26z�
capture_27
�
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�
capture_24
�
capture_25
�
capture_26
�
capture_27B�
/__inference_sequential_14_layer_call_fn_2148803!conv2d_reparameterization_2_input"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_4z�	capture_5z�	capture_6z�	capture_7z�
capture_24z�
capture_25z�
capture_26z�
capture_27
�
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�
capture_24
�
capture_25
�
capture_26
�
capture_27B�
/__inference_sequential_14_layer_call_fn_2149247inputs"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_4z�	capture_5z�	capture_6z�	capture_7z�
capture_24z�
capture_25z�
capture_26z�
capture_27
�
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�
capture_24
�
capture_25
�
capture_26
�
capture_27B�
/__inference_sequential_14_layer_call_fn_2149312inputs"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_4z�	capture_5z�	capture_6z�	capture_7z�
capture_24z�
capture_25z�
capture_26z�
capture_27
�
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�
capture_24
�
capture_25
�
capture_26
�
capture_27B�
J__inference_sequential_14_layer_call_and_return_conditional_losses_2148365!conv2d_reparameterization_2_input"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_4z�	capture_5z�	capture_6z�	capture_7z�
capture_24z�
capture_25z�
capture_26z�
capture_27
�
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�
capture_24
�
capture_25
�
capture_26
�
capture_27B�
J__inference_sequential_14_layer_call_and_return_conditional_losses_2148492!conv2d_reparameterization_2_input"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_4z�	capture_5z�	capture_6z�	capture_7z�
capture_24z�
capture_25z�
capture_26z�
capture_27
�
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�
capture_24
�
capture_25
�
capture_26
�
capture_27B�
J__inference_sequential_14_layer_call_and_return_conditional_losses_2149702inputs"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_4z�	capture_5z�	capture_6z�	capture_7z�
capture_24z�
capture_25z�
capture_26z�
capture_27
�
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�
capture_24
�
capture_25
�
capture_26
�
capture_27B�
J__inference_sequential_14_layer_call_and_return_conditional_losses_2150057inputs"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_4z�	capture_5z�	capture_6z�	capture_7z�
capture_24z�
capture_25z�
capture_26z�
capture_27
!J	
Const_7jtf.TrackableConstant
!J	
Const_6jtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
J
Constjtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�
�	capture_4
�	capture_5
�	capture_6
�	capture_7
�
capture_24
�
capture_25
�
capture_26
�
capture_27B�
%__inference_signature_wrapper_2149182!conv2d_reparameterization_2_input"�
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
 z�	capture_4z�	capture_5z�	capture_6z�	capture_7z�
capture_24z�
capture_25z�
capture_26z�
capture_27
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
�
�	capture_4
�	capture_5
�	capture_6
�	capture_7B�
=__inference_conv2d_reparameterization_2_layer_call_fn_2150080inputs"�
���
FullArgSpec
args�

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
annotations� *
 z�	capture_4z�	capture_5z�	capture_6z�	capture_7
�
�	capture_4
�	capture_5
�	capture_6
�	capture_7B�
X__inference_conv2d_reparameterization_2_layer_call_and_return_conditional_losses_2150225inputs"�
���
FullArgSpec
args�

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
annotations� *
 z�	capture_4z�	capture_5z�	capture_6z�	capture_7
J
%_loc
�_scale
�_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
3
�_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
J
)_loc
�_scale
�_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
3
�_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
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
�B�
2__inference_max_pooling2d_47_layer_call_fn_2150230inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
M__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_2150235inputs"�
���
FullArgSpec
args�

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
annotations� *
 
 "
trackable_list_wrapper
'
90"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
5__inference_depthwise_conv2d_13_layer_call_fn_2150244inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
P__inference_depthwise_conv2d_13_layer_call_and_return_conditional_losses_2150258inputs"�
���
FullArgSpec
args�

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
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

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
annotations� *
 
�2��
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
2__inference_max_pooling2d_48_layer_call_fn_2150263inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
M__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_2150268inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
,__inference_dropout_36_layer_call_fn_2150273inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_dropout_36_layer_call_fn_2150278inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_36_layer_call_and_return_conditional_losses_2150290inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_36_layer_call_and_return_conditional_losses_2150295inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
'
90"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_conv2d_33_layer_call_fn_2150304inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
F__inference_conv2d_33_layer_call_and_return_conditional_losses_2150316inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
2__inference_max_pooling2d_49_layer_call_fn_2150321inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
M__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_2150326inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
,__inference_dropout_37_layer_call_fn_2150331inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_dropout_37_layer_call_fn_2150336inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_37_layer_call_and_return_conditional_losses_2150348inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_37_layer_call_and_return_conditional_losses_2150353inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
'
90"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_conv2d_34_layer_call_fn_2150362inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
F__inference_conv2d_34_layer_call_and_return_conditional_losses_2150374inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
2__inference_max_pooling2d_50_layer_call_fn_2150379inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
M__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_2150384inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
,__inference_dropout_38_layer_call_fn_2150389inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_dropout_38_layer_call_fn_2150394inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_38_layer_call_and_return_conditional_losses_2150406inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_38_layer_call_and_return_conditional_losses_2150411inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
'
90"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_conv2d_35_layer_call_fn_2150420inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
F__inference_conv2d_35_layer_call_and_return_conditional_losses_2150432inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
2__inference_max_pooling2d_51_layer_call_fn_2150437inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
M__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_2150442inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
,__inference_dropout_39_layer_call_fn_2150447inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_dropout_39_layer_call_fn_2150452inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_39_layer_call_and_return_conditional_losses_2150464inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_39_layer_call_and_return_conditional_losses_2150469inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
'
90"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_conv2d_36_layer_call_fn_2150478inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
F__inference_conv2d_36_layer_call_and_return_conditional_losses_2150490inputs"�
���
FullArgSpec
args�

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
annotations� *
 
 "
trackable_list_wrapper
'
90"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_conv2d_37_layer_call_fn_2150499inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
F__inference_conv2d_37_layer_call_and_return_conditional_losses_2150511inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
2__inference_max_pooling2d_52_layer_call_fn_2150516inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
M__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_2150521inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
,__inference_dropout_40_layer_call_fn_2150526inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_dropout_40_layer_call_fn_2150531inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_40_layer_call_and_return_conditional_losses_2150543inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_40_layer_call_and_return_conditional_losses_2150548inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
+__inference_flatten_7_layer_call_fn_2150553inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
F__inference_flatten_7_layer_call_and_return_conditional_losses_2150559inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�
�	capture_4
�	capture_5
�	capture_6
�	capture_7B�
<__inference_dense_reparameterization_2_layer_call_fn_2150582inputs"�
���
FullArgSpec
args�

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
annotations� *
 z�	capture_4z�	capture_5z�	capture_6z�	capture_7
�
�	capture_4
�	capture_5
�	capture_6
�	capture_7B�
W__inference_dense_reparameterization_2_layer_call_and_return_conditional_losses_2150726inputs"�
���
FullArgSpec
args�

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
annotations� *
 z�	capture_4z�	capture_5z�	capture_6z�	capture_7
K
	�_loc
�_scale
�_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
3
�_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
K
	�_loc
�_scale
�_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
3
�_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
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
�B�
7__inference_one_hot_categorical_7_layer_call_fn_2150733inputs"�
���
FullArgSpec
args�

jinputs
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
7__inference_one_hot_categorical_7_layer_call_fn_2150740inputs"�
���
FullArgSpec
args�

jinputs
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
R__inference_one_hot_categorical_7_layer_call_and_return_conditional_losses_2150751inputs"�
���
FullArgSpec
args�

jinputs
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
R__inference_one_hot_categorical_7_layer_call_and_return_conditional_losses_2150762inputs"�
���
FullArgSpec
args�

jinputs
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
O:M 27Adam/m/conv2d_reparameterization_2/kernel_posterior_loc
O:M 27Adam/v/conv2d_reparameterization_2/kernel_posterior_loc
_:] 2GAdam/m/conv2d_reparameterization_2/kernel_posterior_untransformed_scale
_:] 2GAdam/v/conv2d_reparameterization_2/kernel_posterior_untransformed_scale
A:? 25Adam/m/conv2d_reparameterization_2/bias_posterior_loc
A:? 25Adam/v/conv2d_reparameterization_2/bias_posterior_loc
Q:O 2EAdam/m/conv2d_reparameterization_2/bias_posterior_untransformed_scale
Q:O 2EAdam/v/conv2d_reparameterization_2/bias_posterior_untransformed_scale
C:A 2+Adam/m/depthwise_conv2d_13/depthwise_kernel
C:A 2+Adam/v/depthwise_conv2d_13/depthwise_kernel
,:*�2Adam/m/depthwise_conv2d_13/bias
,:*�2Adam/v/depthwise_conv2d_13/bias
0:.�@2Adam/m/conv2d_33/kernel
0:.�@2Adam/v/conv2d_33/kernel
!:@2Adam/m/conv2d_33/bias
!:@2Adam/v/conv2d_33/bias
/:-@@2Adam/m/conv2d_34/kernel
/:-@@2Adam/v/conv2d_34/kernel
!:@2Adam/m/conv2d_34/bias
!:@2Adam/v/conv2d_34/bias
0:.@�2Adam/m/conv2d_35/kernel
0:.@�2Adam/v/conv2d_35/kernel
": �2Adam/m/conv2d_35/bias
": �2Adam/v/conv2d_35/bias
1:/��2Adam/m/conv2d_36/kernel
1:/��2Adam/v/conv2d_36/kernel
": �2Adam/m/conv2d_36/bias
": �2Adam/v/conv2d_36/bias
1:/��2Adam/m/conv2d_37/kernel
1:/��2Adam/v/conv2d_37/kernel
": �2Adam/m/conv2d_37/bias
": �2Adam/v/conv2d_37/bias
G:E	�26Adam/m/dense_reparameterization_2/kernel_posterior_loc
G:E	�26Adam/v/dense_reparameterization_2/kernel_posterior_loc
W:U	�2FAdam/m/dense_reparameterization_2/kernel_posterior_untransformed_scale
W:U	�2FAdam/v/dense_reparameterization_2/kernel_posterior_untransformed_scale
@:>24Adam/m/dense_reparameterization_2/bias_posterior_loc
@:>24Adam/v/dense_reparameterization_2/bias_posterior_loc
P:N2DAdam/m/dense_reparameterization_2/bias_posterior_untransformed_scale
P:N2DAdam/v/dense_reparameterization_2/bias_posterior_untransformed_scale
9
&_pretransformed_input"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
9
*_pretransformed_input"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
:
�_pretransformed_input"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
�_pretransformed_input"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
"__inference__wrapped_model_2147746�,%&)*����:;PQfg|}������������T�Q
J�G
E�B
!conv2d_reparameterization_2_input�����������
� "M�J
H
one_hot_categorical_7/�,
one_hot_categorical_7����������
F__inference_conv2d_33_layer_call_and_return_conditional_losses_2150316tPQ8�5
.�+
)�&
inputs���������%%�
� "4�1
*�'
tensor_0���������%%@
� �
+__inference_conv2d_33_layer_call_fn_2150304iPQ8�5
.�+
)�&
inputs���������%%�
� ")�&
unknown���������%%@�
F__inference_conv2d_34_layer_call_and_return_conditional_losses_2150374sfg7�4
-�*
(�%
inputs���������@
� "4�1
*�'
tensor_0���������@
� �
+__inference_conv2d_34_layer_call_fn_2150362hfg7�4
-�*
(�%
inputs���������@
� ")�&
unknown���������@�
F__inference_conv2d_35_layer_call_and_return_conditional_losses_2150432t|}7�4
-�*
(�%
inputs���������		@
� "5�2
+�(
tensor_0���������		�
� �
+__inference_conv2d_35_layer_call_fn_2150420i|}7�4
-�*
(�%
inputs���������		@
� "*�'
unknown���������		��
F__inference_conv2d_36_layer_call_and_return_conditional_losses_2150490w��8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
+__inference_conv2d_36_layer_call_fn_2150478l��8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
F__inference_conv2d_37_layer_call_and_return_conditional_losses_2150511w��8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
+__inference_conv2d_37_layer_call_fn_2150499l��8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
X__inference_conv2d_reparameterization_2_layer_call_and_return_conditional_losses_2150225�%&)*����9�6
/�,
*�'
inputs�����������
� "`�]
,�)
tensor_0����������� 
-�*
�

tensor_1_0 
�

tensor_1_1 �
=__inference_conv2d_reparameterization_2_layer_call_fn_2150080v%&)*����9�6
/�,
*�'
inputs�����������
� "+�(
unknown����������� �
W__inference_dense_reparameterization_2_layer_call_and_return_conditional_losses_2150726���������0�-
&�#
!�
inputs����������
� "V�S
"�
tensor_0���������
-�*
�

tensor_1_0 
�

tensor_1_1 �
<__inference_dense_reparameterization_2_layer_call_fn_2150582g��������0�-
&�#
!�
inputs����������
� "!�
unknown����������
P__inference_depthwise_conv2d_13_layer_call_and_return_conditional_losses_2150258t:;7�4
-�*
(�%
inputs���������KK 
� "5�2
+�(
tensor_0���������KK�
� �
5__inference_depthwise_conv2d_13_layer_call_fn_2150244i:;7�4
-�*
(�%
inputs���������KK 
� "*�'
unknown���������KK��
G__inference_dropout_36_layer_call_and_return_conditional_losses_2150290u<�9
2�/
)�&
inputs���������%%�
p
� "5�2
+�(
tensor_0���������%%�
� �
G__inference_dropout_36_layer_call_and_return_conditional_losses_2150295u<�9
2�/
)�&
inputs���������%%�
p 
� "5�2
+�(
tensor_0���������%%�
� �
,__inference_dropout_36_layer_call_fn_2150273j<�9
2�/
)�&
inputs���������%%�
p
� "*�'
unknown���������%%��
,__inference_dropout_36_layer_call_fn_2150278j<�9
2�/
)�&
inputs���������%%�
p 
� "*�'
unknown���������%%��
G__inference_dropout_37_layer_call_and_return_conditional_losses_2150348s;�8
1�.
(�%
inputs���������@
p
� "4�1
*�'
tensor_0���������@
� �
G__inference_dropout_37_layer_call_and_return_conditional_losses_2150353s;�8
1�.
(�%
inputs���������@
p 
� "4�1
*�'
tensor_0���������@
� �
,__inference_dropout_37_layer_call_fn_2150331h;�8
1�.
(�%
inputs���������@
p
� ")�&
unknown���������@�
,__inference_dropout_37_layer_call_fn_2150336h;�8
1�.
(�%
inputs���������@
p 
� ")�&
unknown���������@�
G__inference_dropout_38_layer_call_and_return_conditional_losses_2150406s;�8
1�.
(�%
inputs���������		@
p
� "4�1
*�'
tensor_0���������		@
� �
G__inference_dropout_38_layer_call_and_return_conditional_losses_2150411s;�8
1�.
(�%
inputs���������		@
p 
� "4�1
*�'
tensor_0���������		@
� �
,__inference_dropout_38_layer_call_fn_2150389h;�8
1�.
(�%
inputs���������		@
p
� ")�&
unknown���������		@�
,__inference_dropout_38_layer_call_fn_2150394h;�8
1�.
(�%
inputs���������		@
p 
� ")�&
unknown���������		@�
G__inference_dropout_39_layer_call_and_return_conditional_losses_2150464u<�9
2�/
)�&
inputs����������
p
� "5�2
+�(
tensor_0����������
� �
G__inference_dropout_39_layer_call_and_return_conditional_losses_2150469u<�9
2�/
)�&
inputs����������
p 
� "5�2
+�(
tensor_0����������
� �
,__inference_dropout_39_layer_call_fn_2150447j<�9
2�/
)�&
inputs����������
p
� "*�'
unknown�����������
,__inference_dropout_39_layer_call_fn_2150452j<�9
2�/
)�&
inputs����������
p 
� "*�'
unknown�����������
G__inference_dropout_40_layer_call_and_return_conditional_losses_2150543u<�9
2�/
)�&
inputs����������
p
� "5�2
+�(
tensor_0����������
� �
G__inference_dropout_40_layer_call_and_return_conditional_losses_2150548u<�9
2�/
)�&
inputs����������
p 
� "5�2
+�(
tensor_0����������
� �
,__inference_dropout_40_layer_call_fn_2150526j<�9
2�/
)�&
inputs����������
p
� "*�'
unknown�����������
,__inference_dropout_40_layer_call_fn_2150531j<�9
2�/
)�&
inputs����������
p 
� "*�'
unknown�����������
F__inference_flatten_7_layer_call_and_return_conditional_losses_2150559i8�5
.�+
)�&
inputs����������
� "-�*
#� 
tensor_0����������
� �
+__inference_flatten_7_layer_call_fn_2150553^8�5
.�+
)�&
inputs����������
� ""�
unknown�����������
M__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_2150235�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
2__inference_max_pooling2d_47_layer_call_fn_2150230�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
M__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_2150268�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
2__inference_max_pooling2d_48_layer_call_fn_2150263�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
M__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_2150326�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
2__inference_max_pooling2d_49_layer_call_fn_2150321�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
M__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_2150384�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
2__inference_max_pooling2d_50_layer_call_fn_2150379�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
M__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_2150442�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
2__inference_max_pooling2d_51_layer_call_fn_2150437�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
M__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_2150521�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
2__inference_max_pooling2d_52_layer_call_fn_2150516�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
R__inference_one_hot_categorical_7_layer_call_and_return_conditional_losses_2150751o?�<
%�"
 �
inputs���������
�

trainingp",�)
"�
tensor_0���������
� �
R__inference_one_hot_categorical_7_layer_call_and_return_conditional_losses_2150762o?�<
%�"
 �
inputs���������
�

trainingp ",�)
"�
tensor_0���������
� �
7__inference_one_hot_categorical_7_layer_call_fn_2150733�?�<
%�"
 �
inputs���������
�

trainingp"K�H
"�
tensor_0���������
"�
tensor_1����������
7__inference_one_hot_categorical_7_layer_call_fn_2150740�?�<
%�"
 �
inputs���������
�

trainingp "K�H
"�
tensor_0���������
"�
tensor_1����������
J__inference_sequential_14_layer_call_and_return_conditional_losses_2148365�,%&)*����:;PQfg|}������������\�Y
R�O
E�B
!conv2d_reparameterization_2_input�����������
p

 
� "��}
"�
tensor_0���������
W�T
�

tensor_1_0 
�

tensor_1_1 
�

tensor_1_2 
�

tensor_1_3 �
J__inference_sequential_14_layer_call_and_return_conditional_losses_2148492�,%&)*����:;PQfg|}������������\�Y
R�O
E�B
!conv2d_reparameterization_2_input�����������
p 

 
� "��}
"�
tensor_0���������
W�T
�

tensor_1_0 
�

tensor_1_1 
�

tensor_1_2 
�

tensor_1_3 �
J__inference_sequential_14_layer_call_and_return_conditional_losses_2149702�,%&)*����:;PQfg|}������������A�>
7�4
*�'
inputs�����������
p

 
� "��}
"�
tensor_0���������
W�T
�

tensor_1_0 
�

tensor_1_1 
�

tensor_1_2 
�

tensor_1_3 �
J__inference_sequential_14_layer_call_and_return_conditional_losses_2150057�,%&)*����:;PQfg|}������������A�>
7�4
*�'
inputs�����������
p 

 
� "��}
"�
tensor_0���������
W�T
�

tensor_1_0 
�

tensor_1_1 
�

tensor_1_2 
�

tensor_1_3 �
/__inference_sequential_14_layer_call_fn_2148648�,%&)*����:;PQfg|}������������\�Y
R�O
E�B
!conv2d_reparameterization_2_input�����������
p

 
� "!�
unknown����������
/__inference_sequential_14_layer_call_fn_2148803�,%&)*����:;PQfg|}������������\�Y
R�O
E�B
!conv2d_reparameterization_2_input�����������
p 

 
� "!�
unknown����������
/__inference_sequential_14_layer_call_fn_2149247�,%&)*����:;PQfg|}������������A�>
7�4
*�'
inputs�����������
p

 
� "!�
unknown����������
/__inference_sequential_14_layer_call_fn_2149312�,%&)*����:;PQfg|}������������A�>
7�4
*�'
inputs�����������
p 

 
� "!�
unknown����������
%__inference_signature_wrapper_2149182�,%&)*����:;PQfg|}������������y�v
� 
o�l
j
!conv2d_reparameterization_2_inputE�B
!conv2d_reparameterization_2_input�����������"M�J
H
one_hot_categorical_7/�,
one_hot_categorical_7���������