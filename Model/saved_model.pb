Å
¨ý
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
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
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.1.02v2.1.0-rc2-17-ge5bf8de8Ô
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:@*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	@*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:@*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
n
accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameaccumulator
g
accumulator/Read/ReadVariableOpReadVariableOpaccumulator*
_output_shapes
:*
dtype0
r
accumulator_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameaccumulator_1
k
!accumulator_1/Read/ReadVariableOpReadVariableOpaccumulator_1*
_output_shapes
:*
dtype0
r
accumulator_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameaccumulator_2
k
!accumulator_2/Read/ReadVariableOpReadVariableOpaccumulator_2*
_output_shapes
:*
dtype0
r
accumulator_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameaccumulator_3
k
!accumulator_3/Read/ReadVariableOpReadVariableOpaccumulator_3*
_output_shapes
:*
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
t
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
v
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
v
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
dtype0
y
true_positives_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:È*!
shared_nametrue_positives_2
r
$true_positives_2/Read/ReadVariableOpReadVariableOptrue_positives_2*
_output_shapes	
:È*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:È*
dtype0
{
false_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:È*"
shared_namefalse_positives_1
t
%false_positives_1/Read/ReadVariableOpReadVariableOpfalse_positives_1*
_output_shapes	
:È*
dtype0
{
false_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:È*"
shared_namefalse_negatives_1
t
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes	
:È*
dtype0

SGD/dense/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	**
shared_nameSGD/dense/kernel/momentum

-SGD/dense/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense/kernel/momentum*
_output_shapes
:	*
dtype0

SGD/dense/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameSGD/dense/bias/momentum

+SGD/dense/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense/bias/momentum*
_output_shapes	
:*
dtype0

SGD/dense_2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_nameSGD/dense_2/kernel/momentum

/SGD/dense_2/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_2/kernel/momentum*
_output_shapes
:	*
dtype0

SGD/dense_2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameSGD/dense_2/bias/momentum

-SGD/dense_2/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_2/bias/momentum*
_output_shapes	
:*
dtype0

SGD/dense_1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*,
shared_nameSGD/dense_1/kernel/momentum

/SGD/dense_1/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_1/kernel/momentum*
_output_shapes
:	@*
dtype0

SGD/dense_1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameSGD/dense_1/bias/momentum

-SGD/dense_1/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_1/bias/momentum*
_output_shapes
:@*
dtype0

SGD/dense_3/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*,
shared_nameSGD/dense_3/kernel/momentum

/SGD/dense_3/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_3/kernel/momentum*
_output_shapes
:	@*
dtype0

SGD/dense_3/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameSGD/dense_3/bias/momentum

-SGD/dense_3/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_3/bias/momentum*
_output_shapes
:@*
dtype0

SGD/dense_4/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_nameSGD/dense_4/kernel/momentum

/SGD/dense_4/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_4/kernel/momentum*
_output_shapes
:	*
dtype0

SGD/dense_4/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameSGD/dense_4/bias/momentum

-SGD/dense_4/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_4/bias/momentum*
_output_shapes
:*
dtype0

NoOpNoOp
V
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ÅU
value»UB¸U B±U

layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-0
layer-12
layer_with_weights-1
layer-13
layer_with_weights-2
layer-14
layer_with_weights-3
layer-15
layer-16
layer-17
layer-18
layer_with_weights-4
layer-19
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
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
x
_feature_columns

_resources
	variables
trainable_variables
regularization_losses
 	keras_api
x
!_feature_columns
"
_resources
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
h

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
h

3kernel
4bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
h

9kernel
:bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
R
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
R
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
a
G	constants
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
h

Lkernel
Mbias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
à
Riter
	Sdecay
Tlearning_rate
Umomentum'momentumà(momentumá-momentumâ.momentumã3momentumä4momentumå9momentumæ:momentumçLmomentumèMmomentumé
F
'0
(1
-2
.3
34
45
96
:7
L8
M9
F
'0
(1
-2
.3
34
45
96
:7
L8
M9
 

Vlayer_regularization_losses
Wnon_trainable_variables
	variables
trainable_variables
regularization_losses

Xlayers
Ymetrics
 
 
 
 
 
 

Zlayer_regularization_losses

[layers
	variables
trainable_variables
regularization_losses
\non_trainable_variables
]metrics
 
 
 
 
 

^layer_regularization_losses

_layers
#	variables
$trainable_variables
%regularization_losses
`non_trainable_variables
ametrics
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 

blayer_regularization_losses

clayers
)	variables
*trainable_variables
+regularization_losses
dnon_trainable_variables
emetrics
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1

-0
.1
 

flayer_regularization_losses

glayers
/	variables
0trainable_variables
1regularization_losses
hnon_trainable_variables
imetrics
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41

30
41
 

jlayer_regularization_losses

klayers
5	variables
6trainable_variables
7regularization_losses
lnon_trainable_variables
mmetrics
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

90
:1

90
:1
 

nlayer_regularization_losses

olayers
;	variables
<trainable_variables
=regularization_losses
pnon_trainable_variables
qmetrics
 
 
 

rlayer_regularization_losses

slayers
?	variables
@trainable_variables
Aregularization_losses
tnon_trainable_variables
umetrics
 
 
 

vlayer_regularization_losses

wlayers
C	variables
Dtrainable_variables
Eregularization_losses
xnon_trainable_variables
ymetrics
 
 
 
 

zlayer_regularization_losses

{layers
H	variables
Itrainable_variables
Jregularization_losses
|non_trainable_variables
}metrics
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

L0
M1

L0
M1
 

~layer_regularization_losses

layers
N	variables
Otrainable_variables
Pregularization_losses
non_trainable_variables
metrics
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
 
 

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
@
0
1
2
3
4
5
6
7
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
y

thresholds
accumulator
	variables
trainable_variables
regularization_losses
	keras_api
y

thresholds
accumulator
	variables
trainable_variables
regularization_losses
	keras_api
y

thresholds
accumulator
	variables
trainable_variables
regularization_losses
	keras_api
y

thresholds
accumulator
	variables
trainable_variables
 regularization_losses
¡	keras_api


¢total

£count
¤
_fn_kwargs
¥	variables
¦trainable_variables
§regularization_losses
¨	keras_api

©
thresholds
ªtrue_positives
«false_positives
¬	variables
­trainable_variables
®regularization_losses
¯	keras_api

°
thresholds
±true_positives
²false_negatives
³	variables
´trainable_variables
µregularization_losses
¶	keras_api
½
·
thresholds
¸true_positives
¹true_negatives
ºfalse_positives
»false_negatives
¼	variables
½trainable_variables
¾regularization_losses
¿	keras_api
 
[Y
VARIABLE_VALUEaccumulator:keras_api/metrics/0/accumulator/.ATTRIBUTES/VARIABLE_VALUE

0
 
 
¡
 Àlayer_regularization_losses
Álayers
	variables
trainable_variables
regularization_losses
Ânon_trainable_variables
Ãmetrics
 
][
VARIABLE_VALUEaccumulator_1:keras_api/metrics/1/accumulator/.ATTRIBUTES/VARIABLE_VALUE

0
 
 
¡
 Älayer_regularization_losses
Ålayers
	variables
trainable_variables
regularization_losses
Ænon_trainable_variables
Çmetrics
 
][
VARIABLE_VALUEaccumulator_2:keras_api/metrics/2/accumulator/.ATTRIBUTES/VARIABLE_VALUE

0
 
 
¡
 Èlayer_regularization_losses
Élayers
	variables
trainable_variables
regularization_losses
Ênon_trainable_variables
Ëmetrics
 
][
VARIABLE_VALUEaccumulator_3:keras_api/metrics/3/accumulator/.ATTRIBUTES/VARIABLE_VALUE

0
 
 
¡
 Ìlayer_regularization_losses
Ílayers
	variables
trainable_variables
 regularization_losses
Înon_trainable_variables
Ïmetrics
OM
VARIABLE_VALUEtotal4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE
 

¢0
£1
 
 
¡
 Ðlayer_regularization_losses
Ñlayers
¥	variables
¦trainable_variables
§regularization_losses
Ònon_trainable_variables
Ómetrics
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/5/false_positives/.ATTRIBUTES/VARIABLE_VALUE

ª0
«1
 
 
¡
 Ôlayer_regularization_losses
Õlayers
¬	variables
­trainable_variables
®regularization_losses
Önon_trainable_variables
×metrics
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/6/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/6/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

±0
²1
 
 
¡
 Ølayer_regularization_losses
Ùlayers
³	variables
´trainable_variables
µregularization_losses
Únon_trainable_variables
Ûmetrics
 
ca
VARIABLE_VALUEtrue_positives_2=keras_api/metrics/7/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/7/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_positives_1>keras_api/metrics/7/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/7/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
¸0
¹1
º2
»3
 
 
¡
 Ülayer_regularization_losses
Ýlayers
¼	variables
½trainable_variables
¾regularization_losses
Þnon_trainable_variables
ßmetrics
 
 

0
 
 
 

0
 
 
 

0
 
 
 

0
 
 
 

¢0
£1
 
 
 

ª0
«1
 
 
 

±0
²1
 
 
 
 
¸0
¹1
º2
»3
 

VARIABLE_VALUESGD/dense/kernel/momentumYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense/bias/momentumWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_2/kernel/momentumYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_2/bias/momentumWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_1/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_1/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_3/kernel/momentumYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_3/bias/momentumWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_4/kernel/momentumYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_4/bias/momentumWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
s
serving_default_bra_sizePlaceholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
s
serving_default_categoryPlaceholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
s
serving_default_cup_sizePlaceholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
q
serving_default_heightPlaceholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
o
serving_default_hipsPlaceholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
r
serving_default_item_idPlaceholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
q
serving_default_lengthPlaceholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
r
serving_default_qualityPlaceholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
o
serving_default_sizePlaceholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
t
serving_default_user_namePlaceholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_bra_sizeserving_default_categoryserving_default_cup_sizeserving_default_heightserving_default_hipsserving_default_item_idserving_default_lengthserving_default_qualityserving_default_sizeserving_default_user_namedense_2/kerneldense_2/biasdense/kernel
dense/biasdense_3/kerneldense_3/biasdense_1/kerneldense_1/biasdense_4/kerneldense_4/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference_signature_wrapper_23613
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ò
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOpaccumulator/Read/ReadVariableOp!accumulator_1/Read/ReadVariableOp!accumulator_2/Read/ReadVariableOp!accumulator_3/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp$true_positives_2/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp%false_positives_1/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOp-SGD/dense/kernel/momentum/Read/ReadVariableOp+SGD/dense/bias/momentum/Read/ReadVariableOp/SGD/dense_2/kernel/momentum/Read/ReadVariableOp-SGD/dense_2/bias/momentum/Read/ReadVariableOp/SGD/dense_1/kernel/momentum/Read/ReadVariableOp-SGD/dense_1/bias/momentum/Read/ReadVariableOp/SGD/dense_3/kernel/momentum/Read/ReadVariableOp-SGD/dense_3/bias/momentum/Read/ReadVariableOp/SGD/dense_4/kernel/momentum/Read/ReadVariableOp-SGD/dense_4/bias/momentum/Read/ReadVariableOpConst*3
Tin,
*2(	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*'
f"R 
__inference__traced_save_24289
Õ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_2/kerneldense_2/biasdense_1/kerneldense_1/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumaccumulatoraccumulator_1accumulator_2accumulator_3totalcounttrue_positivesfalse_positivestrue_positives_1false_negativestrue_positives_2true_negativesfalse_positives_1false_negatives_1SGD/dense/kernel/momentumSGD/dense/bias/momentumSGD/dense_2/kernel/momentumSGD/dense_2/bias/momentumSGD/dense_1/kernel/momentumSGD/dense_1/bias/momentumSGD/dense_3/kernel/momentumSGD/dense_3/bias/momentumSGD/dense_4/kernel/momentumSGD/dense_4/bias/momentum*2
Tin+
)2'*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__traced_restore_24415öØ


«
0__inference_dense_features_1_layer_call_fn_23969
features_category
features_item_id
features_length
features_quality
features_size
identity
PartitionedCallPartitionedCallfeatures_categoryfeatures_item_idfeatures_lengthfeatures_qualityfeatures_size*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_dense_features_1_layer_call_and_return_conditional_losses_231582
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:1 -
+
_user_specified_namefeatures/category:0,
*
_user_specified_namefeatures/item_id:/+
)
_user_specified_namefeatures/length:0,
*
_user_specified_namefeatures/quality:-)
'
_user_specified_namefeatures/size
ß
b
)__inference_dropout_1_layer_call_fn_24106

inputs
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_233642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Å	
Û
B__inference_dense_3_layer_call_and_return_conditional_losses_24034

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
­I
Ë
__inference__traced_save_24289
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop*
&savev2_accumulator_read_readvariableop,
(savev2_accumulator_1_read_readvariableop,
(savev2_accumulator_2_read_readvariableop,
(savev2_accumulator_3_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_positives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop.
*savev2_false_negatives_read_readvariableop/
+savev2_true_positives_2_read_readvariableop-
)savev2_true_negatives_read_readvariableop0
,savev2_false_positives_1_read_readvariableop0
,savev2_false_negatives_1_read_readvariableop8
4savev2_sgd_dense_kernel_momentum_read_readvariableop6
2savev2_sgd_dense_bias_momentum_read_readvariableop:
6savev2_sgd_dense_2_kernel_momentum_read_readvariableop8
4savev2_sgd_dense_2_bias_momentum_read_readvariableop:
6savev2_sgd_dense_1_kernel_momentum_read_readvariableop8
4savev2_sgd_dense_1_bias_momentum_read_readvariableop:
6savev2_sgd_dense_3_kernel_momentum_read_readvariableop8
4savev2_sgd_dense_3_bias_momentum_read_readvariableop:
6savev2_sgd_dense_4_kernel_momentum_read_readvariableop8
4savev2_sgd_dense_4_bias_momentum_read_readvariableop
savev2_1_const

identity_1¢MergeV2Checkpoints¢SaveV2¢SaveV2_1¥
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_cac73408bbe14f22991dc951f8f09e1e/part2
StringJoin/inputs_1

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename¿
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*Ñ
valueÇBÄ&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/0/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/1/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/2/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/3/accumulator/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/5/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/6/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/6/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/7/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/7/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/7/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/7/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesÔ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop&savev2_accumulator_read_readvariableop(savev2_accumulator_1_read_readvariableop(savev2_accumulator_2_read_readvariableop(savev2_accumulator_3_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_negatives_read_readvariableop+savev2_true_positives_2_read_readvariableop)savev2_true_negatives_read_readvariableop,savev2_false_positives_1_read_readvariableop,savev2_false_negatives_1_read_readvariableop4savev2_sgd_dense_kernel_momentum_read_readvariableop2savev2_sgd_dense_bias_momentum_read_readvariableop6savev2_sgd_dense_2_kernel_momentum_read_readvariableop4savev2_sgd_dense_2_bias_momentum_read_readvariableop6savev2_sgd_dense_1_kernel_momentum_read_readvariableop4savev2_sgd_dense_1_bias_momentum_read_readvariableop6savev2_sgd_dense_3_kernel_momentum_read_readvariableop4savev2_sgd_dense_3_bias_momentum_read_readvariableop6savev2_sgd_dense_4_kernel_momentum_read_readvariableop4savev2_sgd_dense_4_bias_momentum_read_readvariableop"/device:CPU:0*
_output_shapes
 *4
dtypes*
(2&	2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard¬
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1¢
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesÏ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1ã
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¬
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: :	::	::	@:@:	@:@:	:: : : : ::::: : :::::È:È:È:È:	::	::	@:@:	@:@:	:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
ä
Á
#__inference_signature_wrapper_23613
bra_size
category
cup_size

height
hips
item_id

length
quality
size
	user_name#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallbra_sizecup_sizeheighthips	user_namecategoryitem_idlengthqualitysizestatefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__wrapped_model_231132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ó
_input_shapesÁ
¾:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
bra_size:($
"
_user_specified_name
category:($
"
_user_specified_name
cup_size:&"
 
_user_specified_nameheight:$ 

_user_specified_namehips:'#
!
_user_specified_name	item_id:&"
 
_user_specified_namelength:'#
!
_user_specified_name	quality:$ 

_user_specified_namesize:)	%
#
_user_specified_name	user_name
µ5
®
@__inference_model_layer_call_and_return_conditional_losses_23459
bra_size
cup_size

height
hips
	user_name
category
item_id

length
quality
size*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_3_statefulpartitionedcall_args_1*
&dense_3_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall|
dense_features_1/CastCastquality*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features_1/Cast}
dense_features_1/Cast_1Castsize*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features_1/Cast_1¤
 dense_features_1/PartitionedCallPartitionedCallcategoryitem_idlengthdense_features_1/Cast:y:0dense_features_1/Cast_1:y:0*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_dense_features_1_layer_call_and_return_conditional_losses_231582"
 dense_features_1/PartitionedCally
dense_features/CastCastbra_size*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/Cast{
dense_features/Cast_1Castheight*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/Cast_1
dense_features/PartitionedCallPartitionedCalldense_features/Cast:y:0cup_sizedense_features/Cast_1:y:0hips	user_name*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_232022 
dense_features/PartitionedCallÈ
dense_2/StatefulPartitionedCallStatefulPartitionedCall)dense_features_1/PartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_232252!
dense_2/StatefulPartitionedCall¼
dense/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_232482
dense/StatefulPartitionedCallÆ
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0&dense_3_statefulpartitionedcall_args_1&dense_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_232712!
dense_3/StatefulPartitionedCallÄ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_232942!
dense_1/StatefulPartitionedCallÜ
dropout/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_233312
dropout/PartitionedCallâ
dropout_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_233692
dropout_1/PartitionedCall
"tf_op_layer_concat/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_233892$
"tf_op_layer_concat/PartitionedCallÉ
dense_4/StatefulPartitionedCallStatefulPartitionedCall+tf_op_layer_concat/PartitionedCall:output:0&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_234092!
dense_4/StatefulPartitionedCall¤
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ó
_input_shapesÁ
¾:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:( $
"
_user_specified_name
bra_size:($
"
_user_specified_name
cup_size:&"
 
_user_specified_nameheight:$ 

_user_specified_namehips:)%
#
_user_specified_name	user_name:($
"
_user_specified_name
category:'#
!
_user_specified_name	item_id:&"
 
_user_specified_namelength:'#
!
_user_specified_name	quality:$	 

_user_specified_namesize

a
B__inference_dropout_layer_call_and_return_conditional_losses_23326

inputs
identitya
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/random_uniform/max´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02&
$dropout/random_uniform/RandomUniformª
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/subÀ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/random_uniform/mul®
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv¡
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqualp
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/mul
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Castz
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/mul_1e
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:& "
 
_user_specified_nameinputs
Í	
Û
B__inference_dense_4_layer_call_and_return_conditional_losses_23409

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
ï
©
K__inference_dense_features_1_layer_call_and_return_conditional_losses_23158
features

features_1

features_2

features_3

features_4
identity{
quality/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
quality/ExpandDims/dim
quality/ExpandDims
ExpandDims
features_3quality/ExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quality/ExpandDimsi
quality/ShapeShapequality/ExpandDims:output:0*
T0*
_output_shapes
:2
quality/Shape
quality/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
quality/strided_slice/stack
quality/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
quality/strided_slice/stack_1
quality/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
quality/strided_slice/stack_2
quality/strided_sliceStridedSlicequality/Shape:output:0$quality/strided_slice/stack:output:0&quality/strided_slice/stack_1:output:0&quality/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
quality/strided_slicet
quality/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
quality/Reshape/shape/1¦
quality/Reshape/shapePackquality/strided_slice:output:0 quality/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
quality/Reshape/shape
quality/ReshapeReshapequality/ExpandDims:output:0quality/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quality/Reshapeu
size/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
size/ExpandDims/dim
size/ExpandDims
ExpandDims
features_4size/ExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
size/ExpandDims`

size/ShapeShapesize/ExpandDims:output:0*
T0*
_output_shapes
:2

size/Shape~
size/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
size/strided_slice/stack
size/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
size/strided_slice/stack_1
size/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
size/strided_slice/stack_2
size/strided_sliceStridedSlicesize/Shape:output:0!size/strided_slice/stack:output:0#size/strided_slice/stack_1:output:0#size/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
size/strided_slicen
size/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
size/Reshape/shape/1
size/Reshape/shapePacksize/strided_slice:output:0size/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
size/Reshape/shape
size/ReshapeReshapesize/ExpandDims:output:0size/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
size/Reshapee
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
concat/axis
concatConcatV2quality/Reshape:output:0size/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:( $
"
_user_specified_name
features:($
"
_user_specified_name
features:($
"
_user_specified_name
features:($
"
_user_specified_name
features:($
"
_user_specified_name
features
Å5
¹
@__inference_model_layer_call_and_return_conditional_losses_23569

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_3_statefulpartitionedcall_args_1*
&dense_3_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall}
dense_features_1/CastCastinputs_8*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features_1/Cast
dense_features_1/Cast_1Castinputs_9*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features_1/Cast_1§
 dense_features_1/PartitionedCallPartitionedCallinputs_5inputs_6inputs_7dense_features_1/Cast:y:0dense_features_1/Cast_1:y:0*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_dense_features_1_layer_call_and_return_conditional_losses_231582"
 dense_features_1/PartitionedCallw
dense_features/CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/Cast}
dense_features/Cast_1Castinputs_2*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/Cast_1
dense_features/PartitionedCallPartitionedCalldense_features/Cast:y:0inputs_1dense_features/Cast_1:y:0inputs_3inputs_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_232022 
dense_features/PartitionedCallÈ
dense_2/StatefulPartitionedCallStatefulPartitionedCall)dense_features_1/PartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_232252!
dense_2/StatefulPartitionedCall¼
dense/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_232482
dense/StatefulPartitionedCallÆ
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0&dense_3_statefulpartitionedcall_args_1&dense_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_232712!
dense_3/StatefulPartitionedCallÄ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_232942!
dense_1/StatefulPartitionedCallÜ
dropout/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_233312
dropout/PartitionedCallâ
dropout_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_233692
dropout_1/PartitionedCall
"tf_op_layer_concat/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_233892$
"tf_op_layer_concat/PartitionedCallÉ
dense_4/StatefulPartitionedCallStatefulPartitionedCall+tf_op_layer_concat/PartitionedCall:output:0&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_234092!
dense_4/StatefulPartitionedCall¤
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ó
_input_shapesÁ
¾:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs:&	"
 
_user_specified_nameinputs
ì
¦
%__inference_dense_layer_call_fn_23987

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_232482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Å	
Û
B__inference_dense_3_layer_call_and_return_conditional_losses_23271

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Å	
Û
B__inference_dense_1_layer_call_and_return_conditional_losses_23294

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs

w
M__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_23389

inputs
inputs_1
identitye
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*
_cloned(*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
¨
Ù
@__inference_model_layer_call_and_return_conditional_losses_23743
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp}
dense_features_1/CastCastinputs_8*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features_1/Cast
dense_features_1/Cast_1Castinputs_9*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features_1/Cast_1
'dense_features_1/quality/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'dense_features_1/quality/ExpandDims/dim×
#dense_features_1/quality/ExpandDims
ExpandDimsdense_features_1/Cast:y:00dense_features_1/quality/ExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#dense_features_1/quality/ExpandDims
dense_features_1/quality/ShapeShape,dense_features_1/quality/ExpandDims:output:0*
T0*
_output_shapes
:2 
dense_features_1/quality/Shape¦
,dense_features_1/quality/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,dense_features_1/quality/strided_slice/stackª
.dense_features_1/quality/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.dense_features_1/quality/strided_slice/stack_1ª
.dense_features_1/quality/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.dense_features_1/quality/strided_slice/stack_2ø
&dense_features_1/quality/strided_sliceStridedSlice'dense_features_1/quality/Shape:output:05dense_features_1/quality/strided_slice/stack:output:07dense_features_1/quality/strided_slice/stack_1:output:07dense_features_1/quality/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&dense_features_1/quality/strided_slice
(dense_features_1/quality/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(dense_features_1/quality/Reshape/shape/1ê
&dense_features_1/quality/Reshape/shapePack/dense_features_1/quality/strided_slice:output:01dense_features_1/quality/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2(
&dense_features_1/quality/Reshape/shapeà
 dense_features_1/quality/ReshapeReshape,dense_features_1/quality/ExpandDims:output:0/dense_features_1/quality/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 dense_features_1/quality/Reshape
$dense_features_1/size/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2&
$dense_features_1/size/ExpandDims/dimÐ
 dense_features_1/size/ExpandDims
ExpandDimsdense_features_1/Cast_1:y:0-dense_features_1/size/ExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 dense_features_1/size/ExpandDims
dense_features_1/size/ShapeShape)dense_features_1/size/ExpandDims:output:0*
T0*
_output_shapes
:2
dense_features_1/size/Shape 
)dense_features_1/size/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)dense_features_1/size/strided_slice/stack¤
+dense_features_1/size/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+dense_features_1/size/strided_slice/stack_1¤
+dense_features_1/size/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+dense_features_1/size/strided_slice/stack_2æ
#dense_features_1/size/strided_sliceStridedSlice$dense_features_1/size/Shape:output:02dense_features_1/size/strided_slice/stack:output:04dense_features_1/size/strided_slice/stack_1:output:04dense_features_1/size/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#dense_features_1/size/strided_slice
%dense_features_1/size/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%dense_features_1/size/Reshape/shape/1Þ
#dense_features_1/size/Reshape/shapePack,dense_features_1/size/strided_slice:output:0.dense_features_1/size/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2%
#dense_features_1/size/Reshape/shapeÔ
dense_features_1/size/ReshapeReshape)dense_features_1/size/ExpandDims:output:0,dense_features_1/size/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features_1/size/Reshape
dense_features_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
dense_features_1/concat/axisó
dense_features_1/concatConcatV2)dense_features_1/quality/Reshape:output:0&dense_features_1/size/Reshape:output:0%dense_features_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features_1/concaty
dense_features/CastCastinputs_0*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/Cast}
dense_features/Cast_1Castinputs_2*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/Cast_1
&dense_features/bra_size/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&dense_features/bra_size/ExpandDims/dimÒ
"dense_features/bra_size/ExpandDims
ExpandDimsdense_features/Cast:y:0/dense_features/bra_size/ExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"dense_features/bra_size/ExpandDims
dense_features/bra_size/ShapeShape+dense_features/bra_size/ExpandDims:output:0*
T0*
_output_shapes
:2
dense_features/bra_size/Shape¤
+dense_features/bra_size/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+dense_features/bra_size/strided_slice/stack¨
-dense_features/bra_size/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-dense_features/bra_size/strided_slice/stack_1¨
-dense_features/bra_size/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-dense_features/bra_size/strided_slice/stack_2ò
%dense_features/bra_size/strided_sliceStridedSlice&dense_features/bra_size/Shape:output:04dense_features/bra_size/strided_slice/stack:output:06dense_features/bra_size/strided_slice/stack_1:output:06dense_features/bra_size/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%dense_features/bra_size/strided_slice
'dense_features/bra_size/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'dense_features/bra_size/Reshape/shape/1æ
%dense_features/bra_size/Reshape/shapePack.dense_features/bra_size/strided_slice:output:00dense_features/bra_size/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2'
%dense_features/bra_size/Reshape/shapeÜ
dense_features/bra_size/ReshapeReshape+dense_features/bra_size/ExpandDims:output:0.dense_features/bra_size/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
dense_features/bra_size/Reshape
$dense_features/height/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2&
$dense_features/height/ExpandDims/dimÎ
 dense_features/height/ExpandDims
ExpandDimsdense_features/Cast_1:y:0-dense_features/height/ExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 dense_features/height/ExpandDims
dense_features/height/ShapeShape)dense_features/height/ExpandDims:output:0*
T0*
_output_shapes
:2
dense_features/height/Shape 
)dense_features/height/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)dense_features/height/strided_slice/stack¤
+dense_features/height/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+dense_features/height/strided_slice/stack_1¤
+dense_features/height/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+dense_features/height/strided_slice/stack_2æ
#dense_features/height/strided_sliceStridedSlice$dense_features/height/Shape:output:02dense_features/height/strided_slice/stack:output:04dense_features/height/strided_slice/stack_1:output:04dense_features/height/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#dense_features/height/strided_slice
%dense_features/height/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%dense_features/height/Reshape/shape/1Þ
#dense_features/height/Reshape/shapePack,dense_features/height/strided_slice:output:0.dense_features/height/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2%
#dense_features/height/Reshape/shapeÔ
dense_features/height/ReshapeReshape)dense_features/height/ExpandDims:output:0,dense_features/height/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/height/Reshape
dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
dense_features/concat/axisì
dense_features/concatConcatV2(dense_features/bra_size/Reshape:output:0&dense_features/height/Reshape:output:0#dense_features/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/concat¦
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp¦
dense_2/MatMulMatMul dense_features_1/concat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul¥
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¢
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/Relu 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldense_features/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Relu¦
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_3/MatMul¤
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_3/BiasAdd/ReadVariableOp¡
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_3/Relu¦
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp¡
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_1/Reluq
dropout/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/dropout/ratex
dropout/dropout/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/Shape
"dropout/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"dropout/dropout/random_uniform/min
"dropout/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"dropout/dropout/random_uniform/maxÌ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02.
,dropout/dropout/random_uniform/RandomUniformÊ
"dropout/dropout/random_uniform/subSub+dropout/dropout/random_uniform/max:output:0+dropout/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2$
"dropout/dropout/random_uniform/subà
"dropout/dropout/random_uniform/mulMul5dropout/dropout/random_uniform/RandomUniform:output:0&dropout/dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2$
"dropout/dropout/random_uniform/mulÎ
dropout/dropout/random_uniformAdd&dropout/dropout/random_uniform/mul:z:0+dropout/dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
dropout/dropout/random_uniforms
dropout/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/dropout/sub/x
dropout/dropout/subSubdropout/dropout/sub/x:output:0dropout/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/dropout/sub{
dropout/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/dropout/truediv/x
dropout/dropout/truedivRealDiv"dropout/dropout/truediv/x:output:0dropout/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/dropout/truedivÁ
dropout/dropout/GreaterEqualGreaterEqual"dropout/dropout/random_uniform:z:0dropout/dropout/rate:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/dropout/GreaterEqual
dropout/dropout/mulMuldense_1/Relu:activations:0dropout/dropout/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/dropout/mul
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/dropout/Cast
dropout/dropout/mul_1Muldropout/dropout/mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/dropout/mul_1u
dropout_1/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_1/dropout/rate|
dropout_1/dropout/ShapeShapedense_3/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape
$dropout_1/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$dropout_1/dropout/random_uniform/min
$dropout_1/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$dropout_1/dropout/random_uniform/maxÒ
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype020
.dropout_1/dropout/random_uniform/RandomUniformÒ
$dropout_1/dropout/random_uniform/subSub-dropout_1/dropout/random_uniform/max:output:0-dropout_1/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2&
$dropout_1/dropout/random_uniform/subè
$dropout_1/dropout/random_uniform/mulMul7dropout_1/dropout/random_uniform/RandomUniform:output:0(dropout_1/dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$dropout_1/dropout/random_uniform/mulÖ
 dropout_1/dropout/random_uniformAdd(dropout_1/dropout/random_uniform/mul:z:0-dropout_1/dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 dropout_1/dropout/random_uniformw
dropout_1/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout_1/dropout/sub/x
dropout_1/dropout/subSub dropout_1/dropout/sub/x:output:0dropout_1/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout_1/dropout/sub
dropout_1/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout_1/dropout/truediv/x£
dropout_1/dropout/truedivRealDiv$dropout_1/dropout/truediv/x:output:0dropout_1/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout_1/dropout/truedivÉ
dropout_1/dropout/GreaterEqualGreaterEqual$dropout_1/dropout/random_uniform:z:0dropout_1/dropout/rate:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
dropout_1/dropout/GreaterEqual¢
dropout_1/dropout/mulMuldense_3/Relu:activations:0dropout_1/dropout/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_1/dropout/mul
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_1/dropout/Cast¢
dropout_1/dropout/mul_1Muldropout_1/dropout/mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_1/dropout/mul_1
tf_op_layer_concat/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2 
tf_op_layer_concat/concat/axisî
tf_op_layer_concat/concatConcatV2dropout/dropout/mul_1:z:0dropout_1/dropout/mul_1:z:0'tf_op_layer_concat/concat/axis:output:0*
N*
T0*
_cloned(*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf_op_layer_concat/concat¦
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_4/MatMul/ReadVariableOp§
dense_4/MatMulMatMul"tf_op_layer_concat/concat:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/MatMul¤
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp¡
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/BiasAddy
dense_4/SoftmaxSoftmaxdense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/Softmax®
IdentityIdentitydense_4/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ó
_input_shapesÁ
¾:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1:($
"
_user_specified_name
inputs/2:($
"
_user_specified_name
inputs/3:($
"
_user_specified_name
inputs/4:($
"
_user_specified_name
inputs/5:($
"
_user_specified_name
inputs/6:($
"
_user_specified_name
inputs/7:($
"
_user_specified_name
inputs/8:(	$
"
_user_specified_name
inputs/9
º
Æ
K__inference_dense_features_1_layer_call_and_return_conditional_losses_23960
features_category
features_item_id
features_length
features_quality
features_size
identity{
quality/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
quality/ExpandDims/dim
quality/ExpandDims
ExpandDimsfeatures_qualityquality/ExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quality/ExpandDimsi
quality/ShapeShapequality/ExpandDims:output:0*
T0*
_output_shapes
:2
quality/Shape
quality/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
quality/strided_slice/stack
quality/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
quality/strided_slice/stack_1
quality/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
quality/strided_slice/stack_2
quality/strided_sliceStridedSlicequality/Shape:output:0$quality/strided_slice/stack:output:0&quality/strided_slice/stack_1:output:0&quality/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
quality/strided_slicet
quality/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
quality/Reshape/shape/1¦
quality/Reshape/shapePackquality/strided_slice:output:0 quality/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
quality/Reshape/shape
quality/ReshapeReshapequality/ExpandDims:output:0quality/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quality/Reshapeu
size/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
size/ExpandDims/dim
size/ExpandDims
ExpandDimsfeatures_sizesize/ExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
size/ExpandDims`

size/ShapeShapesize/ExpandDims:output:0*
T0*
_output_shapes
:2

size/Shape~
size/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
size/strided_slice/stack
size/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
size/strided_slice/stack_1
size/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
size/strided_slice/stack_2
size/strided_sliceStridedSlicesize/Shape:output:0!size/strided_slice/stack:output:0#size/strided_slice/stack_1:output:0#size/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
size/strided_slicen
size/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
size/Reshape/shape/1
size/Reshape/shapePacksize/strided_slice:output:0size/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
size/Reshape/shape
size/ReshapeReshapesize/ExpandDims:output:0size/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
size/Reshapee
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
concat/axis
concatConcatV2quality/Reshape:output:0size/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:1 -
+
_user_specified_namefeatures/category:0,
*
_user_specified_namefeatures/item_id:/+
)
_user_specified_namefeatures/length:0,
*
_user_specified_namefeatures/quality:-)
'
_user_specified_namefeatures/size

Ã
%__inference_model_layer_call_fn_23582
bra_size
cup_size

height
hips
	user_name
category
item_id

length
quality
size#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallbra_sizecup_sizeheighthips	user_namecategoryitem_idlengthqualitysizestatefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_235692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ó
_input_shapesÁ
¾:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
bra_size:($
"
_user_specified_name
cup_size:&"
 
_user_specified_nameheight:$ 

_user_specified_namehips:)%
#
_user_specified_name	user_name:($
"
_user_specified_name
category:'#
!
_user_specified_name	item_id:&"
 
_user_specified_namelength:'#
!
_user_specified_name	quality:$	 

_user_specified_namesize
É	
Û
B__inference_dense_2_layer_call_and_return_conditional_losses_23998

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Ó
E
)__inference_dropout_1_layer_call_fn_24111

inputs
identity¬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_233692
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:& "
 
_user_specified_nameinputs
ð
¨
'__inference_dense_2_layer_call_fn_24005

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_232252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
¿8
ÿ
@__inference_model_layer_call_and_return_conditional_losses_23508

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_3_statefulpartitionedcall_args_1*
&dense_3_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall}
dense_features_1/CastCastinputs_8*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features_1/Cast
dense_features_1/Cast_1Castinputs_9*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features_1/Cast_1§
 dense_features_1/PartitionedCallPartitionedCallinputs_5inputs_6inputs_7dense_features_1/Cast:y:0dense_features_1/Cast_1:y:0*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_dense_features_1_layer_call_and_return_conditional_losses_231582"
 dense_features_1/PartitionedCallw
dense_features/CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/Cast}
dense_features/Cast_1Castinputs_2*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/Cast_1
dense_features/PartitionedCallPartitionedCalldense_features/Cast:y:0inputs_1dense_features/Cast_1:y:0inputs_3inputs_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_232022 
dense_features/PartitionedCallÈ
dense_2/StatefulPartitionedCallStatefulPartitionedCall)dense_features_1/PartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_232252!
dense_2/StatefulPartitionedCall¼
dense/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_232482
dense/StatefulPartitionedCallÆ
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0&dense_3_statefulpartitionedcall_args_1&dense_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_232712!
dense_3/StatefulPartitionedCallÄ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_232942!
dense_1/StatefulPartitionedCallô
dropout/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_233262!
dropout/StatefulPartitionedCall
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_233642#
!dropout_1/StatefulPartitionedCall«
"tf_op_layer_concat/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_233892$
"tf_op_layer_concat/PartitionedCallÉ
dense_4/StatefulPartitionedCallStatefulPartitionedCall+tf_op_layer_concat/PartitionedCall:output:0&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_234092!
dense_4/StatefulPartitionedCallê
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ó
_input_shapesÁ
¾:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs:&	"
 
_user_specified_nameinputs

Î
!__inference__traced_restore_24415
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_2_kernel#
assignvariableop_3_dense_2_bias%
!assignvariableop_4_dense_1_kernel#
assignvariableop_5_dense_1_bias%
!assignvariableop_6_dense_3_kernel#
assignvariableop_7_dense_3_bias%
!assignvariableop_8_dense_4_kernel#
assignvariableop_9_dense_4_bias 
assignvariableop_10_sgd_iter!
assignvariableop_11_sgd_decay)
%assignvariableop_12_sgd_learning_rate$
 assignvariableop_13_sgd_momentum#
assignvariableop_14_accumulator%
!assignvariableop_15_accumulator_1%
!assignvariableop_16_accumulator_2%
!assignvariableop_17_accumulator_3
assignvariableop_18_total
assignvariableop_19_count&
"assignvariableop_20_true_positives'
#assignvariableop_21_false_positives(
$assignvariableop_22_true_positives_1'
#assignvariableop_23_false_negatives(
$assignvariableop_24_true_positives_2&
"assignvariableop_25_true_negatives)
%assignvariableop_26_false_positives_1)
%assignvariableop_27_false_negatives_11
-assignvariableop_28_sgd_dense_kernel_momentum/
+assignvariableop_29_sgd_dense_bias_momentum3
/assignvariableop_30_sgd_dense_2_kernel_momentum1
-assignvariableop_31_sgd_dense_2_bias_momentum3
/assignvariableop_32_sgd_dense_1_kernel_momentum1
-assignvariableop_33_sgd_dense_1_bias_momentum3
/assignvariableop_34_sgd_dense_3_kernel_momentum1
-assignvariableop_35_sgd_dense_3_bias_momentum3
/assignvariableop_36_sgd_dense_4_kernel_momentum1
-assignvariableop_37_sgd_dense_4_bias_momentum
identity_39¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¢	RestoreV2¢RestoreV2_1Å
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*Ñ
valueÇBÄ&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/0/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/1/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/2/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/3/accumulator/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/5/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/6/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/6/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/7/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/7/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/7/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/7/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesÚ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesì
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*®
_output_shapes
::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_2_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_2_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_1_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_1_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_4_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_4_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0	*
_output_shapes
:2
Identity_10
AssignVariableOp_10AssignVariableOpassignvariableop_10_sgd_iterIdentity_10:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11
AssignVariableOp_11AssignVariableOpassignvariableop_11_sgd_decayIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12
AssignVariableOp_12AssignVariableOp%assignvariableop_12_sgd_learning_rateIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13
AssignVariableOp_13AssignVariableOp assignvariableop_13_sgd_momentumIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14
AssignVariableOp_14AssignVariableOpassignvariableop_14_accumulatorIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15
AssignVariableOp_15AssignVariableOp!assignvariableop_15_accumulator_1Identity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16
AssignVariableOp_16AssignVariableOp!assignvariableop_16_accumulator_2Identity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17
AssignVariableOp_17AssignVariableOp!assignvariableop_17_accumulator_3Identity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19
AssignVariableOp_19AssignVariableOpassignvariableop_19_countIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20
AssignVariableOp_20AssignVariableOp"assignvariableop_20_true_positivesIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21
AssignVariableOp_21AssignVariableOp#assignvariableop_21_false_positivesIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22
AssignVariableOp_22AssignVariableOp$assignvariableop_22_true_positives_1Identity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23
AssignVariableOp_23AssignVariableOp#assignvariableop_23_false_negativesIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24
AssignVariableOp_24AssignVariableOp$assignvariableop_24_true_positives_2Identity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25
AssignVariableOp_25AssignVariableOp"assignvariableop_25_true_negativesIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26
AssignVariableOp_26AssignVariableOp%assignvariableop_26_false_positives_1Identity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27
AssignVariableOp_27AssignVariableOp%assignvariableop_27_false_negatives_1Identity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28¦
AssignVariableOp_28AssignVariableOp-assignvariableop_28_sgd_dense_kernel_momentumIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29¤
AssignVariableOp_29AssignVariableOp+assignvariableop_29_sgd_dense_bias_momentumIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30¨
AssignVariableOp_30AssignVariableOp/assignvariableop_30_sgd_dense_2_kernel_momentumIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31¦
AssignVariableOp_31AssignVariableOp-assignvariableop_31_sgd_dense_2_bias_momentumIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32¨
AssignVariableOp_32AssignVariableOp/assignvariableop_32_sgd_dense_1_kernel_momentumIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33¦
AssignVariableOp_33AssignVariableOp-assignvariableop_33_sgd_dense_1_bias_momentumIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34¨
AssignVariableOp_34AssignVariableOp/assignvariableop_34_sgd_dense_3_kernel_momentumIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35¦
AssignVariableOp_35AssignVariableOp-assignvariableop_35_sgd_dense_3_bias_momentumIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36¨
AssignVariableOp_36AssignVariableOp/assignvariableop_36_sgd_dense_4_kernel_momentumIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37¦
AssignVariableOp_37AssignVariableOp-assignvariableop_37_sgd_dense_4_bias_momentumIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37¨
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesÄ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp¢
Identity_38Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_38¯
Identity_39IdentityIdentity_38:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_39"#
identity_39Identity_39:output:0*¯
_input_shapes
: ::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_37AssignVariableOp_372(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
Å	
Û
B__inference_dense_1_layer_call_and_return_conditional_losses_24016

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Ï
C
'__inference_dropout_layer_call_fn_24076

inputs
identityª
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_233312
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:& "
 
_user_specified_nameinputs
­
Ð
%__inference_model_layer_call_fn_23891
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_235692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ó
_input_shapesÁ
¾:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1:($
"
_user_specified_name
inputs/2:($
"
_user_specified_name
inputs/3:($
"
_user_specified_name
inputs/4:($
"
_user_specified_name
inputs/5:($
"
_user_specified_name
inputs/6:($
"
_user_specified_name
inputs/7:($
"
_user_specified_name
inputs/8:(	$
"
_user_specified_name
inputs/9
ï
¨
'__inference_dense_3_layer_call_fn_24041

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_232712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs

b
D__inference_dropout_1_layer_call_and_return_conditional_losses_24101

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:& "
 
_user_specified_nameinputs

a
B__inference_dropout_layer_call_and_return_conditional_losses_24061

inputs
identitya
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/random_uniform/max´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02&
$dropout/random_uniform/RandomUniformª
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/subÀ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/random_uniform/mul®
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv¡
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqualp
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/mul
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Castz
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/mul_1e
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:& "
 
_user_specified_nameinputs

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_23364

inputs
identitya
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/random_uniform/max´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02&
$dropout/random_uniform/RandomUniformª
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/subÀ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/random_uniform/mul®
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv¡
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqualp
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/mul
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Castz
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/mul_1e
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:& "
 
_user_specified_nameinputs
É
§
I__inference_dense_features_layer_call_and_return_conditional_losses_23202
features

features_1

features_2

features_3

features_4
identity}
bra_size/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
bra_size/ExpandDims/dim
bra_size/ExpandDims
ExpandDimsfeatures bra_size/ExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bra_size/ExpandDimsl
bra_size/ShapeShapebra_size/ExpandDims:output:0*
T0*
_output_shapes
:2
bra_size/Shape
bra_size/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
bra_size/strided_slice/stack
bra_size/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
bra_size/strided_slice/stack_1
bra_size/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
bra_size/strided_slice/stack_2
bra_size/strided_sliceStridedSlicebra_size/Shape:output:0%bra_size/strided_slice/stack:output:0'bra_size/strided_slice/stack_1:output:0'bra_size/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
bra_size/strided_slicev
bra_size/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
bra_size/Reshape/shape/1ª
bra_size/Reshape/shapePackbra_size/strided_slice:output:0!bra_size/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
bra_size/Reshape/shape 
bra_size/ReshapeReshapebra_size/ExpandDims:output:0bra_size/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bra_size/Reshapey
height/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
height/ExpandDims/dim
height/ExpandDims
ExpandDims
features_2height/ExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
height/ExpandDimsf
height/ShapeShapeheight/ExpandDims:output:0*
T0*
_output_shapes
:2
height/Shape
height/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
height/strided_slice/stack
height/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
height/strided_slice/stack_1
height/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
height/strided_slice/stack_2
height/strided_sliceStridedSliceheight/Shape:output:0#height/strided_slice/stack:output:0%height/strided_slice/stack_1:output:0%height/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
height/strided_slicer
height/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
height/Reshape/shape/1¢
height/Reshape/shapePackheight/strided_slice:output:0height/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
height/Reshape/shape
height/ReshapeReshapeheight/ExpandDims:output:0height/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
height/Reshapee
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
concat/axis¡
concatConcatV2bra_size/Reshape:output:0height/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:( $
"
_user_specified_name
features:($
"
_user_specified_name
features:($
"
_user_specified_name
features:($
"
_user_specified_name
features:($
"
_user_specified_name
features

Ã
%__inference_model_layer_call_fn_23521
bra_size
cup_size

height
hips
	user_name
category
item_id

length
quality
size#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallbra_sizecup_sizeheighthips	user_namecategoryitem_idlengthqualitysizestatefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_235082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ó
_input_shapesÁ
¾:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
bra_size:($
"
_user_specified_name
cup_size:&"
 
_user_specified_nameheight:$ 

_user_specified_namehips:)%
#
_user_specified_name	user_name:($
"
_user_specified_name
category:'#
!
_user_specified_name	item_id:&"
 
_user_specified_namelength:'#
!
_user_specified_name	quality:$	 

_user_specified_namesize
Ç	
Ù
@__inference_dense_layer_call_and_return_conditional_losses_23248

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
ï
¨
'__inference_dense_1_layer_call_fn_24023

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_232942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs

Ç
I__inference_dense_features_layer_call_and_return_conditional_losses_23921
features_bra_size
features_cup_size
features_height
features_hips
features_user_name
identity}
bra_size/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
bra_size/ExpandDims/dim
bra_size/ExpandDims
ExpandDimsfeatures_bra_size bra_size/ExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bra_size/ExpandDimsl
bra_size/ShapeShapebra_size/ExpandDims:output:0*
T0*
_output_shapes
:2
bra_size/Shape
bra_size/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
bra_size/strided_slice/stack
bra_size/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
bra_size/strided_slice/stack_1
bra_size/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
bra_size/strided_slice/stack_2
bra_size/strided_sliceStridedSlicebra_size/Shape:output:0%bra_size/strided_slice/stack:output:0'bra_size/strided_slice/stack_1:output:0'bra_size/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
bra_size/strided_slicev
bra_size/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
bra_size/Reshape/shape/1ª
bra_size/Reshape/shapePackbra_size/strided_slice:output:0!bra_size/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
bra_size/Reshape/shape 
bra_size/ReshapeReshapebra_size/ExpandDims:output:0bra_size/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bra_size/Reshapey
height/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
height/ExpandDims/dim
height/ExpandDims
ExpandDimsfeatures_heightheight/ExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
height/ExpandDimsf
height/ShapeShapeheight/ExpandDims:output:0*
T0*
_output_shapes
:2
height/Shape
height/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
height/strided_slice/stack
height/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
height/strided_slice/stack_1
height/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
height/strided_slice/stack_2
height/strided_sliceStridedSliceheight/Shape:output:0#height/strided_slice/stack:output:0%height/strided_slice/stack_1:output:0%height/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
height/strided_slicer
height/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
height/Reshape/shape/1¢
height/Reshape/shapePackheight/strided_slice:output:0height/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
height/Reshape/shape
height/ReshapeReshapeheight/ExpandDims:output:0height/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
height/Reshapee
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
concat/axis¡
concatConcatV2bra_size/Reshape:output:0height/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:1 -
+
_user_specified_namefeatures/bra_size:1-
+
_user_specified_namefeatures/cup_size:/+
)
_user_specified_namefeatures/height:-)
'
_user_specified_namefeatures/hips:2.
,
_user_specified_namefeatures/user_name
¦
¤
 __inference__wrapped_model_23113
bra_size
cup_size

height
hips
	user_name
category
item_id

length
quality
size0
,model_dense_2_matmul_readvariableop_resource1
-model_dense_2_biasadd_readvariableop_resource.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource0
,model_dense_3_matmul_readvariableop_resource1
-model_dense_3_biasadd_readvariableop_resource0
,model_dense_1_matmul_readvariableop_resource1
-model_dense_1_biasadd_readvariableop_resource0
,model_dense_4_matmul_readvariableop_resource1
-model_dense_4_biasadd_readvariableop_resource
identity¢"model/dense/BiasAdd/ReadVariableOp¢!model/dense/MatMul/ReadVariableOp¢$model/dense_1/BiasAdd/ReadVariableOp¢#model/dense_1/MatMul/ReadVariableOp¢$model/dense_2/BiasAdd/ReadVariableOp¢#model/dense_2/MatMul/ReadVariableOp¢$model/dense_3/BiasAdd/ReadVariableOp¢#model/dense_3/MatMul/ReadVariableOp¢$model/dense_4/BiasAdd/ReadVariableOp¢#model/dense_4/MatMul/ReadVariableOp
model/dense_features_1/CastCastquality*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_features_1/Cast
model/dense_features_1/Cast_1Castsize*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_features_1/Cast_1©
-model/dense_features_1/quality/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2/
-model/dense_features_1/quality/ExpandDims/dimï
)model/dense_features_1/quality/ExpandDims
ExpandDimsmodel/dense_features_1/Cast:y:06model/dense_features_1/quality/ExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)model/dense_features_1/quality/ExpandDims®
$model/dense_features_1/quality/ShapeShape2model/dense_features_1/quality/ExpandDims:output:0*
T0*
_output_shapes
:2&
$model/dense_features_1/quality/Shape²
2model/dense_features_1/quality/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2model/dense_features_1/quality/strided_slice/stack¶
4model/dense_features_1/quality/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4model/dense_features_1/quality/strided_slice/stack_1¶
4model/dense_features_1/quality/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4model/dense_features_1/quality/strided_slice/stack_2
,model/dense_features_1/quality/strided_sliceStridedSlice-model/dense_features_1/quality/Shape:output:0;model/dense_features_1/quality/strided_slice/stack:output:0=model/dense_features_1/quality/strided_slice/stack_1:output:0=model/dense_features_1/quality/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,model/dense_features_1/quality/strided_slice¢
.model/dense_features_1/quality/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :20
.model/dense_features_1/quality/Reshape/shape/1
,model/dense_features_1/quality/Reshape/shapePack5model/dense_features_1/quality/strided_slice:output:07model/dense_features_1/quality/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2.
,model/dense_features_1/quality/Reshape/shapeø
&model/dense_features_1/quality/ReshapeReshape2model/dense_features_1/quality/ExpandDims:output:05model/dense_features_1/quality/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&model/dense_features_1/quality/Reshape£
*model/dense_features_1/size/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2,
*model/dense_features_1/size/ExpandDims/dimè
&model/dense_features_1/size/ExpandDims
ExpandDims!model/dense_features_1/Cast_1:y:03model/dense_features_1/size/ExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&model/dense_features_1/size/ExpandDims¥
!model/dense_features_1/size/ShapeShape/model/dense_features_1/size/ExpandDims:output:0*
T0*
_output_shapes
:2#
!model/dense_features_1/size/Shape¬
/model/dense_features_1/size/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/model/dense_features_1/size/strided_slice/stack°
1model/dense_features_1/size/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1model/dense_features_1/size/strided_slice/stack_1°
1model/dense_features_1/size/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1model/dense_features_1/size/strided_slice/stack_2
)model/dense_features_1/size/strided_sliceStridedSlice*model/dense_features_1/size/Shape:output:08model/dense_features_1/size/strided_slice/stack:output:0:model/dense_features_1/size/strided_slice/stack_1:output:0:model/dense_features_1/size/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)model/dense_features_1/size/strided_slice
+model/dense_features_1/size/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+model/dense_features_1/size/Reshape/shape/1ö
)model/dense_features_1/size/Reshape/shapePack2model/dense_features_1/size/strided_slice:output:04model/dense_features_1/size/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2+
)model/dense_features_1/size/Reshape/shapeì
#model/dense_features_1/size/ReshapeReshape/model/dense_features_1/size/ExpandDims:output:02model/dense_features_1/size/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model/dense_features_1/size/Reshape
"model/dense_features_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"model/dense_features_1/concat/axis
model/dense_features_1/concatConcatV2/model/dense_features_1/quality/Reshape:output:0,model/dense_features_1/size/Reshape:output:0+model/dense_features_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_features_1/concat
model/dense_features/CastCastbra_size*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_features/Cast
model/dense_features/Cast_1Castheight*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_features/Cast_1§
,model/dense_features/bra_size/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2.
,model/dense_features/bra_size/ExpandDims/dimê
(model/dense_features/bra_size/ExpandDims
ExpandDimsmodel/dense_features/Cast:y:05model/dense_features/bra_size/ExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(model/dense_features/bra_size/ExpandDims«
#model/dense_features/bra_size/ShapeShape1model/dense_features/bra_size/ExpandDims:output:0*
T0*
_output_shapes
:2%
#model/dense_features/bra_size/Shape°
1model/dense_features/bra_size/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1model/dense_features/bra_size/strided_slice/stack´
3model/dense_features/bra_size/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3model/dense_features/bra_size/strided_slice/stack_1´
3model/dense_features/bra_size/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3model/dense_features/bra_size/strided_slice/stack_2
+model/dense_features/bra_size/strided_sliceStridedSlice,model/dense_features/bra_size/Shape:output:0:model/dense_features/bra_size/strided_slice/stack:output:0<model/dense_features/bra_size/strided_slice/stack_1:output:0<model/dense_features/bra_size/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+model/dense_features/bra_size/strided_slice 
-model/dense_features/bra_size/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-model/dense_features/bra_size/Reshape/shape/1þ
+model/dense_features/bra_size/Reshape/shapePack4model/dense_features/bra_size/strided_slice:output:06model/dense_features/bra_size/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2-
+model/dense_features/bra_size/Reshape/shapeô
%model/dense_features/bra_size/ReshapeReshape1model/dense_features/bra_size/ExpandDims:output:04model/dense_features/bra_size/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%model/dense_features/bra_size/Reshape£
*model/dense_features/height/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2,
*model/dense_features/height/ExpandDims/dimæ
&model/dense_features/height/ExpandDims
ExpandDimsmodel/dense_features/Cast_1:y:03model/dense_features/height/ExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&model/dense_features/height/ExpandDims¥
!model/dense_features/height/ShapeShape/model/dense_features/height/ExpandDims:output:0*
T0*
_output_shapes
:2#
!model/dense_features/height/Shape¬
/model/dense_features/height/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/model/dense_features/height/strided_slice/stack°
1model/dense_features/height/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1model/dense_features/height/strided_slice/stack_1°
1model/dense_features/height/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1model/dense_features/height/strided_slice/stack_2
)model/dense_features/height/strided_sliceStridedSlice*model/dense_features/height/Shape:output:08model/dense_features/height/strided_slice/stack:output:0:model/dense_features/height/strided_slice/stack_1:output:0:model/dense_features/height/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)model/dense_features/height/strided_slice
+model/dense_features/height/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+model/dense_features/height/Reshape/shape/1ö
)model/dense_features/height/Reshape/shapePack2model/dense_features/height/strided_slice:output:04model/dense_features/height/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2+
)model/dense_features/height/Reshape/shapeì
#model/dense_features/height/ReshapeReshape/model/dense_features/height/ExpandDims:output:02model/dense_features/height/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model/dense_features/height/Reshape
 model/dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 model/dense_features/concat/axis
model/dense_features/concatConcatV2.model/dense_features/bra_size/Reshape:output:0,model/dense_features/height/Reshape:output:0)model/dense_features/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_features/concat¸
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02%
#model/dense_2/MatMul/ReadVariableOp¾
model/dense_2/MatMulMatMul&model/dense_features_1/concat:output:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_2/MatMul·
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02&
$model/dense_2/BiasAdd/ReadVariableOpº
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_2/BiasAdd
model/dense_2/ReluRelumodel/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_2/Relu²
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!model/dense/MatMul/ReadVariableOp¶
model/dense/MatMulMatMul$model/dense_features/concat:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense/MatMul±
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"model/dense/BiasAdd/ReadVariableOp²
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense/BiasAdd}
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense/Relu¸
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02%
#model/dense_3/MatMul/ReadVariableOp·
model/dense_3/MatMulMatMul model/dense_2/Relu:activations:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model/dense_3/MatMul¶
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$model/dense_3/BiasAdd/ReadVariableOp¹
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model/dense_3/BiasAdd
model/dense_3/ReluRelumodel/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model/dense_3/Relu¸
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02%
#model/dense_1/MatMul/ReadVariableOpµ
model/dense_1/MatMulMatMulmodel/dense/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model/dense_1/MatMul¶
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp¹
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model/dense_1/BiasAdd
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model/dense_1/Relu
model/dropout/IdentityIdentity model/dense_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model/dropout/Identity
model/dropout_1/IdentityIdentity model/dense_3/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model/dropout_1/Identity
$model/tf_op_layer_concat/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2&
$model/tf_op_layer_concat/concat/axis
model/tf_op_layer_concat/concatConcatV2model/dropout/Identity:output:0!model/dropout_1/Identity:output:0-model/tf_op_layer_concat/concat/axis:output:0*
N*
T0*
_cloned(*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
model/tf_op_layer_concat/concat¸
#model/dense_4/MatMul/ReadVariableOpReadVariableOp,model_dense_4_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02%
#model/dense_4/MatMul/ReadVariableOp¿
model/dense_4/MatMulMatMul(model/tf_op_layer_concat/concat:output:0+model/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_4/MatMul¶
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_4/BiasAdd/ReadVariableOp¹
model/dense_4/BiasAddBiasAddmodel/dense_4/MatMul:product:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_4/BiasAdd
model/dense_4/SoftmaxSoftmaxmodel/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_4/Softmaxð
IdentityIdentitymodel/dense_4/Softmax:softmax:0#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp$^model/dense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ó
_input_shapesÁ
¾:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2J
#model/dense_3/MatMul/ReadVariableOp#model/dense_3/MatMul/ReadVariableOp2L
$model/dense_4/BiasAdd/ReadVariableOp$model/dense_4/BiasAdd/ReadVariableOp2J
#model/dense_4/MatMul/ReadVariableOp#model/dense_4/MatMul/ReadVariableOp:( $
"
_user_specified_name
bra_size:($
"
_user_specified_name
cup_size:&"
 
_user_specified_nameheight:$ 

_user_specified_namehips:)%
#
_user_specified_name	user_name:($
"
_user_specified_name
category:'#
!
_user_specified_name	item_id:&"
 
_user_specified_namelength:'#
!
_user_specified_name	quality:$	 

_user_specified_namesize
­
Ð
%__inference_model_layer_call_fn_23867
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_235082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ó
_input_shapesÁ
¾:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1:($
"
_user_specified_name
inputs/2:($
"
_user_specified_name
inputs/3:($
"
_user_specified_name
inputs/4:($
"
_user_specified_name
inputs/5:($
"
_user_specified_name
inputs/6:($
"
_user_specified_name
inputs/7:($
"
_user_specified_name
inputs/8:(	$
"
_user_specified_name
inputs/9
Ç	
Ù
@__inference_dense_layer_call_and_return_conditional_losses_23980

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Û
`
'__inference_dropout_layer_call_fn_24071

inputs
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_233262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_24096

inputs
identitya
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/random_uniform/max´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02&
$dropout/random_uniform/RandomUniformª
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/subÀ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/random_uniform/mul®
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv¡
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqualp
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/mul
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Castz
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/mul_1e
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:& "
 
_user_specified_nameinputs

`
B__inference_dropout_layer_call_and_return_conditional_losses_24066

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:& "
 
_user_specified_nameinputs

`
B__inference_dropout_layer_call_and_return_conditional_losses_23331

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:& "
 
_user_specified_nameinputs
Ã
^
2__inference_tf_op_layer_concat_layer_call_fn_24124
inputs_0
inputs_1
identityÃ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_233892
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
ï
¨
'__inference_dense_4_layer_call_fn_24142

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_234092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
É	
Û
B__inference_dense_2_layer_call_and_return_conditional_losses_23225

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
õ
Ù
@__inference_model_layer_call_and_return_conditional_losses_23843
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp}
dense_features_1/CastCastinputs_8*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features_1/Cast
dense_features_1/Cast_1Castinputs_9*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features_1/Cast_1
'dense_features_1/quality/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'dense_features_1/quality/ExpandDims/dim×
#dense_features_1/quality/ExpandDims
ExpandDimsdense_features_1/Cast:y:00dense_features_1/quality/ExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#dense_features_1/quality/ExpandDims
dense_features_1/quality/ShapeShape,dense_features_1/quality/ExpandDims:output:0*
T0*
_output_shapes
:2 
dense_features_1/quality/Shape¦
,dense_features_1/quality/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,dense_features_1/quality/strided_slice/stackª
.dense_features_1/quality/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.dense_features_1/quality/strided_slice/stack_1ª
.dense_features_1/quality/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.dense_features_1/quality/strided_slice/stack_2ø
&dense_features_1/quality/strided_sliceStridedSlice'dense_features_1/quality/Shape:output:05dense_features_1/quality/strided_slice/stack:output:07dense_features_1/quality/strided_slice/stack_1:output:07dense_features_1/quality/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&dense_features_1/quality/strided_slice
(dense_features_1/quality/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(dense_features_1/quality/Reshape/shape/1ê
&dense_features_1/quality/Reshape/shapePack/dense_features_1/quality/strided_slice:output:01dense_features_1/quality/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2(
&dense_features_1/quality/Reshape/shapeà
 dense_features_1/quality/ReshapeReshape,dense_features_1/quality/ExpandDims:output:0/dense_features_1/quality/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 dense_features_1/quality/Reshape
$dense_features_1/size/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2&
$dense_features_1/size/ExpandDims/dimÐ
 dense_features_1/size/ExpandDims
ExpandDimsdense_features_1/Cast_1:y:0-dense_features_1/size/ExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 dense_features_1/size/ExpandDims
dense_features_1/size/ShapeShape)dense_features_1/size/ExpandDims:output:0*
T0*
_output_shapes
:2
dense_features_1/size/Shape 
)dense_features_1/size/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)dense_features_1/size/strided_slice/stack¤
+dense_features_1/size/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+dense_features_1/size/strided_slice/stack_1¤
+dense_features_1/size/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+dense_features_1/size/strided_slice/stack_2æ
#dense_features_1/size/strided_sliceStridedSlice$dense_features_1/size/Shape:output:02dense_features_1/size/strided_slice/stack:output:04dense_features_1/size/strided_slice/stack_1:output:04dense_features_1/size/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#dense_features_1/size/strided_slice
%dense_features_1/size/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%dense_features_1/size/Reshape/shape/1Þ
#dense_features_1/size/Reshape/shapePack,dense_features_1/size/strided_slice:output:0.dense_features_1/size/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2%
#dense_features_1/size/Reshape/shapeÔ
dense_features_1/size/ReshapeReshape)dense_features_1/size/ExpandDims:output:0,dense_features_1/size/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features_1/size/Reshape
dense_features_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
dense_features_1/concat/axisó
dense_features_1/concatConcatV2)dense_features_1/quality/Reshape:output:0&dense_features_1/size/Reshape:output:0%dense_features_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features_1/concaty
dense_features/CastCastinputs_0*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/Cast}
dense_features/Cast_1Castinputs_2*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/Cast_1
&dense_features/bra_size/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&dense_features/bra_size/ExpandDims/dimÒ
"dense_features/bra_size/ExpandDims
ExpandDimsdense_features/Cast:y:0/dense_features/bra_size/ExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"dense_features/bra_size/ExpandDims
dense_features/bra_size/ShapeShape+dense_features/bra_size/ExpandDims:output:0*
T0*
_output_shapes
:2
dense_features/bra_size/Shape¤
+dense_features/bra_size/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+dense_features/bra_size/strided_slice/stack¨
-dense_features/bra_size/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-dense_features/bra_size/strided_slice/stack_1¨
-dense_features/bra_size/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-dense_features/bra_size/strided_slice/stack_2ò
%dense_features/bra_size/strided_sliceStridedSlice&dense_features/bra_size/Shape:output:04dense_features/bra_size/strided_slice/stack:output:06dense_features/bra_size/strided_slice/stack_1:output:06dense_features/bra_size/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%dense_features/bra_size/strided_slice
'dense_features/bra_size/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'dense_features/bra_size/Reshape/shape/1æ
%dense_features/bra_size/Reshape/shapePack.dense_features/bra_size/strided_slice:output:00dense_features/bra_size/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2'
%dense_features/bra_size/Reshape/shapeÜ
dense_features/bra_size/ReshapeReshape+dense_features/bra_size/ExpandDims:output:0.dense_features/bra_size/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
dense_features/bra_size/Reshape
$dense_features/height/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2&
$dense_features/height/ExpandDims/dimÎ
 dense_features/height/ExpandDims
ExpandDimsdense_features/Cast_1:y:0-dense_features/height/ExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 dense_features/height/ExpandDims
dense_features/height/ShapeShape)dense_features/height/ExpandDims:output:0*
T0*
_output_shapes
:2
dense_features/height/Shape 
)dense_features/height/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)dense_features/height/strided_slice/stack¤
+dense_features/height/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+dense_features/height/strided_slice/stack_1¤
+dense_features/height/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+dense_features/height/strided_slice/stack_2æ
#dense_features/height/strided_sliceStridedSlice$dense_features/height/Shape:output:02dense_features/height/strided_slice/stack:output:04dense_features/height/strided_slice/stack_1:output:04dense_features/height/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#dense_features/height/strided_slice
%dense_features/height/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%dense_features/height/Reshape/shape/1Þ
#dense_features/height/Reshape/shapePack,dense_features/height/strided_slice:output:0.dense_features/height/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2%
#dense_features/height/Reshape/shapeÔ
dense_features/height/ReshapeReshape)dense_features/height/ExpandDims:output:0,dense_features/height/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/height/Reshape
dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
dense_features/concat/axisì
dense_features/concatConcatV2(dense_features/bra_size/Reshape:output:0&dense_features/height/Reshape:output:0#dense_features/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/concat¦
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp¦
dense_2/MatMulMatMul dense_features_1/concat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul¥
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¢
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/Relu 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldense_features/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Relu¦
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_3/MatMul¤
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_3/BiasAdd/ReadVariableOp¡
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_3/Relu¦
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp¡
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_1/Relu~
dropout/IdentityIdentitydense_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Identity
dropout_1/IdentityIdentitydense_3/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_1/Identity
tf_op_layer_concat/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2 
tf_op_layer_concat/concat/axisî
tf_op_layer_concat/concatConcatV2dropout/Identity:output:0dropout_1/Identity:output:0'tf_op_layer_concat/concat/axis:output:0*
N*
T0*
_cloned(*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf_op_layer_concat/concat¦
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_4/MatMul/ReadVariableOp§
dense_4/MatMulMatMul"tf_op_layer_concat/concat:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/MatMul¤
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp¡
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/BiasAddy
dense_4/SoftmaxSoftmaxdense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/Softmax®
IdentityIdentitydense_4/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ó
_input_shapesÁ
¾:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1:($
"
_user_specified_name
inputs/2:($
"
_user_specified_name
inputs/3:($
"
_user_specified_name
inputs/4:($
"
_user_specified_name
inputs/5:($
"
_user_specified_name
inputs/6:($
"
_user_specified_name
inputs/7:($
"
_user_specified_name
inputs/8:(	$
"
_user_specified_name
inputs/9

y
M__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_24118
inputs_0
inputs_1
identitye
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*
_cloned(*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
Í	
Û
B__inference_dense_4_layer_call_and_return_conditional_losses_24135

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
 
¬
.__inference_dense_features_layer_call_fn_23930
features_bra_size
features_cup_size
features_height
features_hips
features_user_name
identity
PartitionedCallPartitionedCallfeatures_bra_sizefeatures_cup_sizefeatures_heightfeatures_hipsfeatures_user_name*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_232022
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:1 -
+
_user_specified_namefeatures/bra_size:1-
+
_user_specified_namefeatures/cup_size:/+
)
_user_specified_namefeatures/height:-)
'
_user_specified_namefeatures/hips:2.
,
_user_specified_namefeatures/user_name
¯8
ô
@__inference_model_layer_call_and_return_conditional_losses_23422
bra_size
cup_size

height
hips
	user_name
category
item_id

length
quality
size*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_3_statefulpartitionedcall_args_1*
&dense_3_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall|
dense_features_1/CastCastquality*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features_1/Cast}
dense_features_1/Cast_1Castsize*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features_1/Cast_1¤
 dense_features_1/PartitionedCallPartitionedCallcategoryitem_idlengthdense_features_1/Cast:y:0dense_features_1/Cast_1:y:0*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_dense_features_1_layer_call_and_return_conditional_losses_231582"
 dense_features_1/PartitionedCally
dense_features/CastCastbra_size*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/Cast{
dense_features/Cast_1Castheight*

DstT0*

SrcT0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_features/Cast_1
dense_features/PartitionedCallPartitionedCalldense_features/Cast:y:0cup_sizedense_features/Cast_1:y:0hips	user_name*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_232022 
dense_features/PartitionedCallÈ
dense_2/StatefulPartitionedCallStatefulPartitionedCall)dense_features_1/PartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_232252!
dense_2/StatefulPartitionedCall¼
dense/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_232482
dense/StatefulPartitionedCallÆ
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0&dense_3_statefulpartitionedcall_args_1&dense_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_232712!
dense_3/StatefulPartitionedCallÄ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_232942!
dense_1/StatefulPartitionedCallô
dropout/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_233262!
dropout/StatefulPartitionedCall
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_233642#
!dropout_1/StatefulPartitionedCall«
"tf_op_layer_concat/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_233892$
"tf_op_layer_concat/PartitionedCallÉ
dense_4/StatefulPartitionedCallStatefulPartitionedCall+tf_op_layer_concat/PartitionedCall:output:0&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_234092!
dense_4/StatefulPartitionedCallê
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ó
_input_shapesÁ
¾:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:( $
"
_user_specified_name
bra_size:($
"
_user_specified_name
cup_size:&"
 
_user_specified_nameheight:$ 

_user_specified_namehips:)%
#
_user_specified_name	user_name:($
"
_user_specified_name
category:'#
!
_user_specified_name	item_id:&"
 
_user_specified_namelength:'#
!
_user_specified_name	quality:$	 

_user_specified_namesize

b
D__inference_dropout_1_layer_call_and_return_conditional_losses_23369

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:& "
 
_user_specified_nameinputs"¯L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¡
serving_default
9
bra_size-
serving_default_bra_size:0ÿÿÿÿÿÿÿÿÿ
9
category-
serving_default_category:0ÿÿÿÿÿÿÿÿÿ
9
cup_size-
serving_default_cup_size:0ÿÿÿÿÿÿÿÿÿ
5
height+
serving_default_height:0ÿÿÿÿÿÿÿÿÿ
1
hips)
serving_default_hips:0ÿÿÿÿÿÿÿÿÿ
7
item_id,
serving_default_item_id:0ÿÿÿÿÿÿÿÿÿ
5
length+
serving_default_length:0ÿÿÿÿÿÿÿÿÿ
7
quality,
serving_default_quality:0ÿÿÿÿÿÿÿÿÿ
1
size)
serving_default_size:0ÿÿÿÿÿÿÿÿÿ
;
	user_name.
serving_default_user_name:0ÿÿÿÿÿÿÿÿÿ;
dense_40
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:²
¯
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-0
layer-12
layer_with_weights-1
layer-13
layer_with_weights-2
layer-14
layer_with_weights-3
layer-15
layer-16
layer-17
layer-18
layer_with_weights-4
layer-19
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
ê__call__
+ë&call_and_return_all_conditional_losses
ì_default_save_signature"Ð
_tf_keras_modelµ{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null], "dtype": "float64", "sparse": false, "ragged": false, "name": "bra_size"}, "name": "bra_size", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null], "dtype": "string", "sparse": false, "ragged": false, "name": "cup_size"}, "name": "cup_size", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null], "dtype": "float64", "sparse": false, "ragged": false, "name": "height"}, "name": "height", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null], "dtype": "string", "sparse": false, "ragged": false, "name": "hips"}, "name": "hips", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null], "dtype": "string", "sparse": false, "ragged": false, "name": "user_name"}, "name": "user_name", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null], "dtype": "string", "sparse": false, "ragged": false, "name": "category"}, "name": "category", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null], "dtype": "string", "sparse": false, "ragged": false, "name": "item_id"}, "name": "item_id", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null], "dtype": "string", "sparse": false, "ragged": false, "name": "length"}, "name": "length", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null], "dtype": "float64", "sparse": false, "ragged": false, "name": "quality"}, "name": "quality", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null], "dtype": "float64", "sparse": false, "ragged": false, "name": "size"}, "name": "size", "inbound_nodes": []}, {"class_name": "DenseFeatures", "config": {"name": "dense_features", "trainable": true, "dtype": "float32", "feature_columns": [{"class_name": "NumericColumn", "config": {"key": "bra_size", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "height", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}]}, "name": "dense_features", "inbound_nodes": [{"height": ["height", 0, 0, {}], "bra_size": ["bra_size", 0, 0, {}], "hips": ["hips", 0, 0, {}], "cup_size": ["cup_size", 0, 0, {}], "user_name": ["user_name", 0, 0, {}]}]}, {"class_name": "DenseFeatures", "config": {"name": "dense_features_1", "trainable": true, "dtype": "float32", "feature_columns": [{"class_name": "NumericColumn", "config": {"key": "quality", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "size", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}]}, "name": "dense_features_1", "inbound_nodes": [{"size": ["size", 0, 0, {}], "quality": ["quality", 0, 0, {}], "item_id": ["item_id", 0, 0, {}], "category": ["category", 0, 0, {}], "length": ["length", 0, 0, {}]}]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dense_features", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_features_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "concat", "trainable": true, "dtype": "float32", "node_def": {"name": "concat", "op": "ConcatV2", "input": ["dropout/Identity", "dropout_1/Identity", "concat/axis"], "attr": {"T": {"type": "DT_FLOAT"}, "Tidx": {"type": "DT_INT32"}, "N": {"i": "2"}}}, "constants": {"2": -1}}, "name": "tf_op_layer_concat", "inbound_nodes": [[["dropout", 0, 0, {}], ["dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["tf_op_layer_concat", 0, 0, {}]]]}], "input_layers": [{"height": ["height", 0, 0], "bra_size": ["bra_size", 0, 0], "hips": ["hips", 0, 0], "cup_size": ["cup_size", 0, 0], "user_name": ["user_name", 0, 0]}, {"size": ["size", 0, 0], "quality": ["quality", 0, 0], "item_id": ["item_id", 0, 0], "category": ["category", 0, 0], "length": ["length", 0, 0]}], "output_layers": [["dense_4", 0, 0]]}, "input_spec": [null, null, null, null, null, null, null, null, null, null], "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null], "dtype": "float64", "sparse": false, "ragged": false, "name": "bra_size"}, "name": "bra_size", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null], "dtype": "string", "sparse": false, "ragged": false, "name": "cup_size"}, "name": "cup_size", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null], "dtype": "float64", "sparse": false, "ragged": false, "name": "height"}, "name": "height", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null], "dtype": "string", "sparse": false, "ragged": false, "name": "hips"}, "name": "hips", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null], "dtype": "string", "sparse": false, "ragged": false, "name": "user_name"}, "name": "user_name", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null], "dtype": "string", "sparse": false, "ragged": false, "name": "category"}, "name": "category", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null], "dtype": "string", "sparse": false, "ragged": false, "name": "item_id"}, "name": "item_id", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null], "dtype": "string", "sparse": false, "ragged": false, "name": "length"}, "name": "length", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null], "dtype": "float64", "sparse": false, "ragged": false, "name": "quality"}, "name": "quality", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null], "dtype": "float64", "sparse": false, "ragged": false, "name": "size"}, "name": "size", "inbound_nodes": []}, {"class_name": "DenseFeatures", "config": {"name": "dense_features", "trainable": true, "dtype": "float32", "feature_columns": [{"class_name": "NumericColumn", "config": {"key": "bra_size", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "height", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}]}, "name": "dense_features", "inbound_nodes": [{"height": ["height", 0, 0, {}], "bra_size": ["bra_size", 0, 0, {}], "hips": ["hips", 0, 0, {}], "cup_size": ["cup_size", 0, 0, {}], "user_name": ["user_name", 0, 0, {}]}]}, {"class_name": "DenseFeatures", "config": {"name": "dense_features_1", "trainable": true, "dtype": "float32", "feature_columns": [{"class_name": "NumericColumn", "config": {"key": "quality", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "size", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}]}, "name": "dense_features_1", "inbound_nodes": [{"size": ["size", 0, 0, {}], "quality": ["quality", 0, 0, {}], "item_id": ["item_id", 0, 0, {}], "category": ["category", 0, 0, {}], "length": ["length", 0, 0, {}]}]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dense_features", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_features_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "concat", "trainable": true, "dtype": "float32", "node_def": {"name": "concat", "op": "ConcatV2", "input": ["dropout/Identity", "dropout_1/Identity", "concat/axis"], "attr": {"T": {"type": "DT_FLOAT"}, "Tidx": {"type": "DT_INT32"}, "N": {"i": "2"}}}, "constants": {"2": -1}}, "name": "tf_op_layer_concat", "inbound_nodes": [[["dropout", 0, 0, {}], ["dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["tf_op_layer_concat", 0, 0, {}]]]}], "input_layers": [{"height": ["height", 0, 0], "bra_size": ["bra_size", 0, 0], "hips": ["hips", 0, 0], "cup_size": ["cup_size", 0, 0], "user_name": ["user_name", 0, 0]}, {"size": ["size", 0, 0], "quality": ["quality", 0, 0], "item_id": ["item_id", 0, 0], "category": ["category", 0, 0], "length": ["length", 0, 0]}], "output_layers": [["dense_4", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [{"class_name": "TruePositives", "config": {"name": "tp", "dtype": "float32", "thresholds": null}}, {"class_name": "FalsePositives", "config": {"name": "fp", "dtype": "float32", "thresholds": null}}, {"class_name": "TrueNegatives", "config": {"name": "tn", "dtype": "float32", "thresholds": null}}, {"class_name": "FalseNegatives", "config": {"name": "fn", "dtype": "float32", "thresholds": null}}, {"class_name": "BinaryAccuracy", "config": {"name": "accuracy", "dtype": "float32", "threshold": 0.5}}, {"class_name": "Precision", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}, {"class_name": "Recall", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}, {"class_name": "AUC", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.10000000149011612, "decay": 0.009999999776482582, "momentum": 0.8999999761581421, "nesterov": false}}}}
"
_tf_keras_input_layerö{"class_name": "InputLayer", "name": "bra_size", "dtype": "float64", "sparse": false, "ragged": false, "batch_input_shape": [null], "config": {"batch_input_shape": [null], "dtype": "float64", "sparse": false, "ragged": false, "name": "bra_size"}}
"
_tf_keras_input_layerô{"class_name": "InputLayer", "name": "cup_size", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": [null], "config": {"batch_input_shape": [null], "dtype": "string", "sparse": false, "ragged": false, "name": "cup_size"}}
"
_tf_keras_input_layerò{"class_name": "InputLayer", "name": "height", "dtype": "float64", "sparse": false, "ragged": false, "batch_input_shape": [null], "config": {"batch_input_shape": [null], "dtype": "float64", "sparse": false, "ragged": false, "name": "height"}}
"
_tf_keras_input_layerì{"class_name": "InputLayer", "name": "hips", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": [null], "config": {"batch_input_shape": [null], "dtype": "string", "sparse": false, "ragged": false, "name": "hips"}}
"
_tf_keras_input_layerö{"class_name": "InputLayer", "name": "user_name", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": [null], "config": {"batch_input_shape": [null], "dtype": "string", "sparse": false, "ragged": false, "name": "user_name"}}
"
_tf_keras_input_layerô{"class_name": "InputLayer", "name": "category", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": [null], "config": {"batch_input_shape": [null], "dtype": "string", "sparse": false, "ragged": false, "name": "category"}}
"
_tf_keras_input_layerò{"class_name": "InputLayer", "name": "item_id", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": [null], "config": {"batch_input_shape": [null], "dtype": "string", "sparse": false, "ragged": false, "name": "item_id"}}
"
_tf_keras_input_layerð{"class_name": "InputLayer", "name": "length", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": [null], "config": {"batch_input_shape": [null], "dtype": "string", "sparse": false, "ragged": false, "name": "length"}}
"
_tf_keras_input_layerô{"class_name": "InputLayer", "name": "quality", "dtype": "float64", "sparse": false, "ragged": false, "batch_input_shape": [null], "config": {"batch_input_shape": [null], "dtype": "float64", "sparse": false, "ragged": false, "name": "quality"}}
"
_tf_keras_input_layerî{"class_name": "InputLayer", "name": "size", "dtype": "float64", "sparse": false, "ragged": false, "batch_input_shape": [null], "config": {"batch_input_shape": [null], "dtype": "float64", "sparse": false, "ragged": false, "name": "size"}}

_feature_columns

_resources
	variables
trainable_variables
regularization_losses
 	keras_api
í__call__
+î&call_and_return_all_conditional_losses"Ï
_tf_keras_layerµ{"class_name": "DenseFeatures", "name": "dense_features", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_features", "trainable": true, "dtype": "float32", "feature_columns": [{"class_name": "NumericColumn", "config": {"key": "bra_size", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "height", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}]}, "_is_feature_layer": true}

!_feature_columns
"
_resources
#	variables
$trainable_variables
%regularization_losses
&	keras_api
ï__call__
+ð&call_and_return_all_conditional_losses"Ð
_tf_keras_layer¶{"class_name": "DenseFeatures", "name": "dense_features_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_features_1", "trainable": true, "dtype": "float32", "feature_columns": [{"class_name": "NumericColumn", "config": {"key": "quality", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "size", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}]}, "_is_feature_layer": true}
ï

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
ñ__call__
+ò&call_and_return_all_conditional_losses"È
_tf_keras_layer®{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}}
ó

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
ó__call__
+ô&call_and_return_all_conditional_losses"Ì
_tf_keras_layer²{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}}
ô

3kernel
4bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
õ__call__
+ö&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}}
ô

9kernel
:bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
÷__call__
+ø&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}}
­
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
ù__call__
+ú&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
±
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
û__call__
+ü&call_and_return_all_conditional_losses" 
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}

G	constants
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
ý__call__
+þ&call_and_return_all_conditional_losses"ã
_tf_keras_layerÉ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_concat", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "concat", "trainable": true, "dtype": "float32", "node_def": {"name": "concat", "op": "ConcatV2", "input": ["dropout/Identity", "dropout_1/Identity", "concat/axis"], "attr": {"T": {"type": "DT_FLOAT"}, "Tidx": {"type": "DT_INT32"}, "N": {"i": "2"}}}, "constants": {"2": -1}}}
ö

Lkernel
Mbias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
ÿ__call__
+&call_and_return_all_conditional_losses"Ï
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
ó
Riter
	Sdecay
Tlearning_rate
Umomentum'momentumà(momentumá-momentumâ.momentumã3momentumä4momentumå9momentumæ:momentumçLmomentumèMmomentumé"
	optimizer
f
'0
(1
-2
.3
34
45
96
:7
L8
M9"
trackable_list_wrapper
f
'0
(1
-2
.3
34
45
96
:7
L8
M9"
trackable_list_wrapper
 "
trackable_list_wrapper
»
Vlayer_regularization_losses
Wnon_trainable_variables
	variables
trainable_variables
regularization_losses

Xlayers
Ymetrics
ê__call__
ì_default_save_signature
+ë&call_and_return_all_conditional_losses
'ë"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
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

Zlayer_regularization_losses

[layers
	variables
trainable_variables
regularization_losses
\non_trainable_variables
]metrics
í__call__
+î&call_and_return_all_conditional_losses
'î"call_and_return_conditional_losses"
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

^layer_regularization_losses

_layers
#	variables
$trainable_variables
%regularization_losses
`non_trainable_variables
ametrics
ï__call__
+ð&call_and_return_all_conditional_losses
'ð"call_and_return_conditional_losses"
_generic_user_object
:	2dense/kernel
:2
dense/bias
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

blayer_regularization_losses

clayers
)	variables
*trainable_variables
+regularization_losses
dnon_trainable_variables
emetrics
ñ__call__
+ò&call_and_return_all_conditional_losses
'ò"call_and_return_conditional_losses"
_generic_user_object
!:	2dense_2/kernel
:2dense_2/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper

flayer_regularization_losses

glayers
/	variables
0trainable_variables
1regularization_losses
hnon_trainable_variables
imetrics
ó__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses"
_generic_user_object
!:	@2dense_1/kernel
:@2dense_1/bias
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper

jlayer_regularization_losses

klayers
5	variables
6trainable_variables
7regularization_losses
lnon_trainable_variables
mmetrics
õ__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
_generic_user_object
!:	@2dense_3/kernel
:@2dense_3/bias
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper

nlayer_regularization_losses

olayers
;	variables
<trainable_variables
=regularization_losses
pnon_trainable_variables
qmetrics
÷__call__
+ø&call_and_return_all_conditional_losses
'ø"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

rlayer_regularization_losses

slayers
?	variables
@trainable_variables
Aregularization_losses
tnon_trainable_variables
umetrics
ù__call__
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

vlayer_regularization_losses

wlayers
C	variables
Dtrainable_variables
Eregularization_losses
xnon_trainable_variables
ymetrics
û__call__
+ü&call_and_return_all_conditional_losses
'ü"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

zlayer_regularization_losses

{layers
H	variables
Itrainable_variables
Jregularization_losses
|non_trainable_variables
}metrics
ý__call__
+þ&call_and_return_all_conditional_losses
'þ"call_and_return_conditional_losses"
_generic_user_object
!:	2dense_4/kernel
:2dense_4/bias
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper

~layer_regularization_losses

layers
N	variables
Otrainable_variables
Pregularization_losses
non_trainable_variables
metrics
ÿ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¶
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
19"
trackable_list_wrapper
`
0
1
2
3
4
5
6
7"
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¡

thresholds
accumulator
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"é
_tf_keras_layerÏ{"class_name": "TruePositives", "name": "tp", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "tp", "dtype": "float32", "thresholds": null}}
¢

thresholds
accumulator
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"ê
_tf_keras_layerÐ{"class_name": "FalsePositives", "name": "fp", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "fp", "dtype": "float32", "thresholds": null}}
¡

thresholds
accumulator
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"é
_tf_keras_layerÏ{"class_name": "TrueNegatives", "name": "tn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "tn", "dtype": "float32", "thresholds": null}}
¢

thresholds
accumulator
	variables
trainable_variables
 regularization_losses
¡	keras_api
__call__
+&call_and_return_all_conditional_losses"ê
_tf_keras_layerÐ{"class_name": "FalseNegatives", "name": "fn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "fn", "dtype": "float32", "thresholds": null}}
²

¢total

£count
¤
_fn_kwargs
¥	variables
¦trainable_variables
§regularization_losses
¨	keras_api
__call__
+&call_and_return_all_conditional_losses"ô
_tf_keras_layerÚ{"class_name": "BinaryAccuracy", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32", "threshold": 0.5}}
å
©
thresholds
ªtrue_positives
«false_positives
¬	variables
­trainable_variables
®regularization_losses
¯	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerú{"class_name": "Precision", "name": "precision", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
Ü
°
thresholds
±true_positives
²false_negatives
³	variables
´trainable_variables
µregularization_losses
¶	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerñ{"class_name": "Recall", "name": "recall", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
$
·
thresholds
¸true_positives
¹true_negatives
ºfalse_positives
»false_negatives
¼	variables
½trainable_variables
¾regularization_losses
¿	keras_api
__call__
+&call_and_return_all_conditional_losses""
_tf_keras_layerí!{"class_name": "AUC", "name": "auc", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}
 "
trackable_list_wrapper
: (2accumulator
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¤
 Àlayer_regularization_losses
Álayers
	variables
trainable_variables
regularization_losses
Ânon_trainable_variables
Ãmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¤
 Älayer_regularization_losses
Ålayers
	variables
trainable_variables
regularization_losses
Ænon_trainable_variables
Çmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¤
 Èlayer_regularization_losses
Élayers
	variables
trainable_variables
regularization_losses
Ênon_trainable_variables
Ëmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¤
 Ìlayer_regularization_losses
Ílayers
	variables
trainable_variables
 regularization_losses
Înon_trainable_variables
Ïmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
¢0
£1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¤
 Ðlayer_regularization_losses
Ñlayers
¥	variables
¦trainable_variables
§regularization_losses
Ònon_trainable_variables
Ómetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
ª0
«1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¤
 Ôlayer_regularization_losses
Õlayers
¬	variables
­trainable_variables
®regularization_losses
Önon_trainable_variables
×metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
±0
²1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¤
 Ølayer_regularization_losses
Ùlayers
³	variables
´trainable_variables
µregularization_losses
Únon_trainable_variables
Ûmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:È (2true_positives
:È (2true_negatives
 :È (2false_positives
 :È (2false_negatives
@
¸0
¹1
º2
»3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¤
 Ülayer_regularization_losses
Ýlayers
¼	variables
½trainable_variables
¾regularization_losses
Þnon_trainable_variables
ßmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
¢0
£1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
ª0
«1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
±0
²1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
¸0
¹1
º2
»3"
trackable_list_wrapper
 "
trackable_list_wrapper
*:(	2SGD/dense/kernel/momentum
$:"2SGD/dense/bias/momentum
,:*	2SGD/dense_2/kernel/momentum
&:$2SGD/dense_2/bias/momentum
,:*	@2SGD/dense_1/kernel/momentum
%:#@2SGD/dense_1/bias/momentum
,:*	@2SGD/dense_3/kernel/momentum
%:#@2SGD/dense_3/bias/momentum
,:*	2SGD/dense_4/kernel/momentum
%:#2SGD/dense_4/bias/momentum
â2ß
%__inference_model_layer_call_fn_23521
%__inference_model_layer_call_fn_23891
%__inference_model_layer_call_fn_23582
%__inference_model_layer_call_fn_23867À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Î2Ë
@__inference_model_layer_call_and_return_conditional_losses_23459
@__inference_model_layer_call_and_return_conditional_losses_23422
@__inference_model_layer_call_and_return_conditional_losses_23843
@__inference_model_layer_call_and_return_conditional_losses_23743À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
÷2ô
 __inference__wrapped_model_23113Ï
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¾¢º
·³

bra_sizeÿÿÿÿÿÿÿÿÿ

cup_sizeÿÿÿÿÿÿÿÿÿ

heightÿÿÿÿÿÿÿÿÿ

hipsÿÿÿÿÿÿÿÿÿ

	user_nameÿÿÿÿÿÿÿÿÿ

categoryÿÿÿÿÿÿÿÿÿ

item_idÿÿÿÿÿÿÿÿÿ

lengthÿÿÿÿÿÿÿÿÿ

qualityÿÿÿÿÿÿÿÿÿ

sizeÿÿÿÿÿÿÿÿÿ
ù2ö
.__inference_dense_features_layer_call_fn_23930Ã
º²¶
FullArgSpec9
args1.
jself

jfeatures
jcols_to_output_tensors
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
I__inference_dense_features_layer_call_and_return_conditional_losses_23921Ã
º²¶
FullArgSpec9
args1.
jself

jfeatures
jcols_to_output_tensors
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
û2ø
0__inference_dense_features_1_layer_call_fn_23969Ã
º²¶
FullArgSpec9
args1.
jself

jfeatures
jcols_to_output_tensors
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
K__inference_dense_features_1_layer_call_and_return_conditional_losses_23960Ã
º²¶
FullArgSpec9
args1.
jself

jfeatures
jcols_to_output_tensors
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_dense_layer_call_fn_23987¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_dense_layer_call_and_return_conditional_losses_23980¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_2_layer_call_fn_24005¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_2_layer_call_and_return_conditional_losses_23998¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_1_layer_call_fn_24023¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_1_layer_call_and_return_conditional_losses_24016¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_3_layer_call_fn_24041¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_3_layer_call_and_return_conditional_losses_24034¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
'__inference_dropout_layer_call_fn_24076
'__inference_dropout_layer_call_fn_24071´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Â2¿
B__inference_dropout_layer_call_and_return_conditional_losses_24066
B__inference_dropout_layer_call_and_return_conditional_losses_24061´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
)__inference_dropout_1_layer_call_fn_24106
)__inference_dropout_1_layer_call_fn_24111´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout_1_layer_call_and_return_conditional_losses_24101
D__inference_dropout_1_layer_call_and_return_conditional_losses_24096´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ü2Ù
2__inference_tf_op_layer_concat_layer_call_fn_24124¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_24118¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_4_layer_call_fn_24142¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_4_layer_call_and_return_conditional_losses_24135¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
~B|
#__inference_signature_wrapper_23613bra_sizecategorycup_sizeheighthipsitem_idlengthqualitysize	user_name
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 ±
 __inference__wrapped_model_23113
-.'(9:34LMÊ¢Æ
¾¢º
·³

bra_sizeÿÿÿÿÿÿÿÿÿ

cup_sizeÿÿÿÿÿÿÿÿÿ

heightÿÿÿÿÿÿÿÿÿ

hipsÿÿÿÿÿÿÿÿÿ

	user_nameÿÿÿÿÿÿÿÿÿ

categoryÿÿÿÿÿÿÿÿÿ

item_idÿÿÿÿÿÿÿÿÿ

lengthÿÿÿÿÿÿÿÿÿ

qualityÿÿÿÿÿÿÿÿÿ

sizeÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
dense_4!
dense_4ÿÿÿÿÿÿÿÿÿ£
B__inference_dense_1_layer_call_and_return_conditional_losses_24016]340¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 {
'__inference_dense_1_layer_call_fn_24023P340¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@£
B__inference_dense_2_layer_call_and_return_conditional_losses_23998]-./¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 {
'__inference_dense_2_layer_call_fn_24005P-./¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
B__inference_dense_3_layer_call_and_return_conditional_losses_24034]9:0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 {
'__inference_dense_3_layer_call_fn_24041P9:0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@£
B__inference_dense_4_layer_call_and_return_conditional_losses_24135]LM0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
'__inference_dense_4_layer_call_fn_24142PLM0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
K__inference_dense_features_1_layer_call_and_return_conditional_losses_23960¾¢
¢
ýªù
3
category'$
features/categoryÿÿÿÿÿÿÿÿÿ
1
item_id&#
features/item_idÿÿÿÿÿÿÿÿÿ
/
length%"
features/lengthÿÿÿÿÿÿÿÿÿ
1
quality&#
features/qualityÿÿÿÿÿÿÿÿÿ
+
size# 
features/sizeÿÿÿÿÿÿÿÿÿ

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 æ
0__inference_dense_features_1_layer_call_fn_23969±¢
¢
ýªù
3
category'$
features/categoryÿÿÿÿÿÿÿÿÿ
1
item_id&#
features/item_idÿÿÿÿÿÿÿÿÿ
/
length%"
features/lengthÿÿÿÿÿÿÿÿÿ
1
quality&#
features/qualityÿÿÿÿÿÿÿÿÿ
+
size# 
features/sizeÿÿÿÿÿÿÿÿÿ

 
ª "ÿÿÿÿÿÿÿÿÿ
I__inference_dense_features_layer_call_and_return_conditional_losses_23921Ä¢
¢
ªÿ
3
bra_size'$
features/bra_sizeÿÿÿÿÿÿÿÿÿ
3
cup_size'$
features/cup_sizeÿÿÿÿÿÿÿÿÿ
/
height%"
features/heightÿÿÿÿÿÿÿÿÿ
+
hips# 
features/hipsÿÿÿÿÿÿÿÿÿ
5
	user_name(%
features/user_nameÿÿÿÿÿÿÿÿÿ

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ê
.__inference_dense_features_layer_call_fn_23930·¢
¢
ªÿ
3
bra_size'$
features/bra_sizeÿÿÿÿÿÿÿÿÿ
3
cup_size'$
features/cup_sizeÿÿÿÿÿÿÿÿÿ
/
height%"
features/heightÿÿÿÿÿÿÿÿÿ
+
hips# 
features/hipsÿÿÿÿÿÿÿÿÿ
5
	user_name(%
features/user_nameÿÿÿÿÿÿÿÿÿ

 
ª "ÿÿÿÿÿÿÿÿÿ¡
@__inference_dense_layer_call_and_return_conditional_losses_23980]'(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 y
%__inference_dense_layer_call_fn_23987P'(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
D__inference_dropout_1_layer_call_and_return_conditional_losses_24096\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¤
D__inference_dropout_1_layer_call_and_return_conditional_losses_24101\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 |
)__inference_dropout_1_layer_call_fn_24106O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@|
)__inference_dropout_1_layer_call_fn_24111O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@¢
B__inference_dropout_layer_call_and_return_conditional_losses_24061\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¢
B__inference_dropout_layer_call_and_return_conditional_losses_24066\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 z
'__inference_dropout_layer_call_fn_24071O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@z
'__inference_dropout_layer_call_fn_24076O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@Í
@__inference_model_layer_call_and_return_conditional_losses_23422
-.'(9:34LMÒ¢Î
Æ¢Â
·³

bra_sizeÿÿÿÿÿÿÿÿÿ

cup_sizeÿÿÿÿÿÿÿÿÿ

heightÿÿÿÿÿÿÿÿÿ

hipsÿÿÿÿÿÿÿÿÿ

	user_nameÿÿÿÿÿÿÿÿÿ

categoryÿÿÿÿÿÿÿÿÿ

item_idÿÿÿÿÿÿÿÿÿ

lengthÿÿÿÿÿÿÿÿÿ

qualityÿÿÿÿÿÿÿÿÿ

sizeÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Í
@__inference_model_layer_call_and_return_conditional_losses_23459
-.'(9:34LMÒ¢Î
Æ¢Â
·³

bra_sizeÿÿÿÿÿÿÿÿÿ

cup_sizeÿÿÿÿÿÿÿÿÿ

heightÿÿÿÿÿÿÿÿÿ

hipsÿÿÿÿÿÿÿÿÿ

	user_nameÿÿÿÿÿÿÿÿÿ

categoryÿÿÿÿÿÿÿÿÿ

item_idÿÿÿÿÿÿÿÿÿ

lengthÿÿÿÿÿÿÿÿÿ

qualityÿÿÿÿÿÿÿÿÿ

sizeÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ú
@__inference_model_layer_call_and_return_conditional_losses_23743
-.'(9:34LMß¢Û
Ó¢Ï
ÄÀ

inputs/0ÿÿÿÿÿÿÿÿÿ

inputs/1ÿÿÿÿÿÿÿÿÿ

inputs/2ÿÿÿÿÿÿÿÿÿ

inputs/3ÿÿÿÿÿÿÿÿÿ

inputs/4ÿÿÿÿÿÿÿÿÿ

inputs/5ÿÿÿÿÿÿÿÿÿ

inputs/6ÿÿÿÿÿÿÿÿÿ

inputs/7ÿÿÿÿÿÿÿÿÿ

inputs/8ÿÿÿÿÿÿÿÿÿ

inputs/9ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ú
@__inference_model_layer_call_and_return_conditional_losses_23843
-.'(9:34LMß¢Û
Ó¢Ï
ÄÀ

inputs/0ÿÿÿÿÿÿÿÿÿ

inputs/1ÿÿÿÿÿÿÿÿÿ

inputs/2ÿÿÿÿÿÿÿÿÿ

inputs/3ÿÿÿÿÿÿÿÿÿ

inputs/4ÿÿÿÿÿÿÿÿÿ

inputs/5ÿÿÿÿÿÿÿÿÿ

inputs/6ÿÿÿÿÿÿÿÿÿ

inputs/7ÿÿÿÿÿÿÿÿÿ

inputs/8ÿÿÿÿÿÿÿÿÿ

inputs/9ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¥
%__inference_model_layer_call_fn_23521û
-.'(9:34LMÒ¢Î
Æ¢Â
·³

bra_sizeÿÿÿÿÿÿÿÿÿ

cup_sizeÿÿÿÿÿÿÿÿÿ

heightÿÿÿÿÿÿÿÿÿ

hipsÿÿÿÿÿÿÿÿÿ

	user_nameÿÿÿÿÿÿÿÿÿ

categoryÿÿÿÿÿÿÿÿÿ

item_idÿÿÿÿÿÿÿÿÿ

lengthÿÿÿÿÿÿÿÿÿ

qualityÿÿÿÿÿÿÿÿÿ

sizeÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¥
%__inference_model_layer_call_fn_23582û
-.'(9:34LMÒ¢Î
Æ¢Â
·³

bra_sizeÿÿÿÿÿÿÿÿÿ

cup_sizeÿÿÿÿÿÿÿÿÿ

heightÿÿÿÿÿÿÿÿÿ

hipsÿÿÿÿÿÿÿÿÿ

	user_nameÿÿÿÿÿÿÿÿÿ

categoryÿÿÿÿÿÿÿÿÿ

item_idÿÿÿÿÿÿÿÿÿ

lengthÿÿÿÿÿÿÿÿÿ

qualityÿÿÿÿÿÿÿÿÿ

sizeÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ²
%__inference_model_layer_call_fn_23867
-.'(9:34LMß¢Û
Ó¢Ï
ÄÀ

inputs/0ÿÿÿÿÿÿÿÿÿ

inputs/1ÿÿÿÿÿÿÿÿÿ

inputs/2ÿÿÿÿÿÿÿÿÿ

inputs/3ÿÿÿÿÿÿÿÿÿ

inputs/4ÿÿÿÿÿÿÿÿÿ

inputs/5ÿÿÿÿÿÿÿÿÿ

inputs/6ÿÿÿÿÿÿÿÿÿ

inputs/7ÿÿÿÿÿÿÿÿÿ

inputs/8ÿÿÿÿÿÿÿÿÿ

inputs/9ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ²
%__inference_model_layer_call_fn_23891
-.'(9:34LMß¢Û
Ó¢Ï
ÄÀ

inputs/0ÿÿÿÿÿÿÿÿÿ

inputs/1ÿÿÿÿÿÿÿÿÿ

inputs/2ÿÿÿÿÿÿÿÿÿ

inputs/3ÿÿÿÿÿÿÿÿÿ

inputs/4ÿÿÿÿÿÿÿÿÿ

inputs/5ÿÿÿÿÿÿÿÿÿ

inputs/6ÿÿÿÿÿÿÿÿÿ

inputs/7ÿÿÿÿÿÿÿÿÿ

inputs/8ÿÿÿÿÿÿÿÿÿ

inputs/9ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
#__inference_signature_wrapper_23613ð
-.'(9:34LM®¢ª
¢ 
¢ª
*
bra_size
bra_sizeÿÿÿÿÿÿÿÿÿ
*
category
categoryÿÿÿÿÿÿÿÿÿ
*
cup_size
cup_sizeÿÿÿÿÿÿÿÿÿ
&
height
heightÿÿÿÿÿÿÿÿÿ
"
hips
hipsÿÿÿÿÿÿÿÿÿ
(
item_id
item_idÿÿÿÿÿÿÿÿÿ
&
length
lengthÿÿÿÿÿÿÿÿÿ
(
quality
qualityÿÿÿÿÿÿÿÿÿ
"
size
sizeÿÿÿÿÿÿÿÿÿ
,
	user_name
	user_nameÿÿÿÿÿÿÿÿÿ"1ª.
,
dense_4!
dense_4ÿÿÿÿÿÿÿÿÿÖ
M__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_24118Z¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ@
"
inputs/1ÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ­
2__inference_tf_op_layer_concat_layer_call_fn_24124wZ¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ@
"
inputs/1ÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ