
E
Placeholder/label_0Placeholder*
dtype0*
shape:	?@
E
Placeholder/label_1Placeholder*
dtype0*
shape:	?@
E
Placeholder/label_2Placeholder*
dtype0*
shape:	?@
E
Placeholder/label_3Placeholder*
dtype0*
shape:	?@
E
Placeholder/label_4Placeholder*
dtype0*
shape:	?@
E
Placeholder/label_5Placeholder*
dtype0*
shape:	?@
U
#Placeholder/old_label_probability_0Placeholder*
dtype0*
shape:	?@
U
#Placeholder/old_label_probability_1Placeholder*
dtype0*
shape:	?@
U
#Placeholder/old_label_probability_2Placeholder*
dtype0*
shape:	?@
U
#Placeholder/old_label_probability_3Placeholder*
dtype0*
shape:	?@
U
#Placeholder/old_label_probability_4Placeholder*
dtype0*
shape:	?@
U
#Placeholder/old_label_probability_5Placeholder*
dtype0*
shape:	?@
I
Placeholder/fc2_label_0Placeholder*
dtype0*
shape:	?@
I
Placeholder/fc2_label_1Placeholder*
dtype0*
shape:	?@
I
Placeholder/fc2_label_2Placeholder*
dtype0*
shape:	?@
I
Placeholder/fc2_label_3Placeholder*
dtype0*
shape:	?@
I
Placeholder/fc2_label_4Placeholder*
dtype0*
shape:	?@
I
Placeholder/fc2_label_5Placeholder*
dtype0*
shape:	?@
D
Placeholder/rewardPlaceholder*
dtype0*
shape:	?@
G
Placeholder/advantagePlaceholder*
dtype0*
shape:	?@
N
Placeholder/fc2_value_resultPlaceholder*
dtype0*
shape:	?@
G
Placeholder/seri_vecPlaceholder*
dtype0*
shape:
?@?
K
Placeholder/weight_list_0Placeholder*
dtype0*
shape:	?@
K
Placeholder/weight_list_1Placeholder*
dtype0*
shape:	?@
K
Placeholder/weight_list_2Placeholder*
dtype0*
shape:	?@
K
Placeholder/weight_list_3Placeholder*
dtype0*
shape:	?@
K
Placeholder/weight_list_4Placeholder*
dtype0*
shape:	?@
K
Placeholder/weight_list_5Placeholder*
dtype0*
shape:	?@
F
SqueezeSqueezePlaceholder/reward*
T0*
squeeze_dims

K
	Squeeze_1SqueezePlaceholder/advantage*
T0*
squeeze_dims

I
	Squeeze_2SqueezePlaceholder/label_0*
T0*
squeeze_dims

I
	Squeeze_3SqueezePlaceholder/label_1*
T0*
squeeze_dims

I
	Squeeze_4SqueezePlaceholder/label_2*
T0*
squeeze_dims

I
	Squeeze_5SqueezePlaceholder/label_3*
T0*
squeeze_dims

I
	Squeeze_6SqueezePlaceholder/label_4*
T0*
squeeze_dims

I
	Squeeze_7SqueezePlaceholder/label_5*
T0*
squeeze_dims

O
	Squeeze_8SqueezePlaceholder/weight_list_0*
T0*
squeeze_dims

O
	Squeeze_9SqueezePlaceholder/weight_list_1*
T0*
squeeze_dims

P

Squeeze_10SqueezePlaceholder/weight_list_2*
T0*
squeeze_dims

P

Squeeze_11SqueezePlaceholder/weight_list_3*
T0*
squeeze_dims

P

Squeeze_12SqueezePlaceholder/weight_list_4*
T0*
squeeze_dims

P

Squeeze_13SqueezePlaceholder/weight_list_5*
T0*
squeeze_dims

:
ConstConst*
dtype0*
valueB"?  T   
9
split/split_dimConst*
dtype0*
value	B :
c
splitSplitVPlaceholder/seri_vecConstsplit/split_dim*
T0*

Tlen0*
	num_split
B
Reshape/shapeConst*
dtype0*
valueB"????T   
A
ReshapeReshapesplit:1Reshape/shape*
T0*
Tshape0
L
Const_1Const*
dtype0*-
value$B""                  
;
split_1/split_dimConst*
dtype0*
value	B :
\
split_1SplitVReshapeConst_1split_1/split_dim*
T0*

Tlen0*
	num_split
S

Squeeze_14SqueezePlaceholder/fc2_value_result*
T0*
squeeze_dims

(
subSubSqueeze
Squeeze_14*
T0

SquareSquaresub*
T0
@
Mean/reduction_indicesConst*
dtype0*
value	B : 
R
MeanMeanSquareMean/reduction_indices*
T0*

Tidx0*
	keep_dims( 
2
mul/xConst*
dtype0*
valueB
 *   ?
 
mulMulmul/xMean*
T0
4
Const_2Const*
dtype0*
valueB
 *    
4
Const_3Const*
dtype0*
valueB
 *    
=
one_hot/on_valueConst*
dtype0*
valueB
 *  ??
>
one_hot/off_valueConst*
dtype0*
valueB
 *    
7
one_hot/depthConst*
dtype0*
value	B :
x
one_hotOneHot	Squeeze_2one_hot/depthone_hot/on_valueone_hot/off_value*
T0*
TI0*
axis?????????
4
sub_1/xConst*
dtype0*
valueB
 *  ??
'
sub_1Subsub_1/xsplit_1*
T0
2
Pow/xConst*
dtype0*
valueB
 *   A
2
Pow/yConst*
dtype0*
valueB
 *  ?A
!
PowPowPow/xPow/y*
T0
!
mul_1Mulsub_1Pow*
T0
5
sub_2SubPlaceholder/fc2_label_0mul_1*
T0
?
Max/reduction_indicesConst*
dtype0*
value	B :
N
MaxMaxsub_2Max/reduction_indices*
T0*

Tidx0*
	keep_dims(
3
sub_3SubPlaceholder/fc2_label_0Max*
T0
4
Pow_1/xConst*
dtype0*
valueB
 *   A
4
Pow_1/yConst*
dtype0*
valueB
 *  ?A
'
Pow_1PowPow_1/xPow_1/y*
T0

NegNegPow_1*
T0
D
clip_by_value/Minimum/yConst*
dtype0*
valueB
 *  ??
I
clip_by_value/MinimumMinimumsub_3clip_by_value/Minimum/y*
T0
=
clip_by_valueMaximumclip_by_value/MinimumNeg*
T0
"
ExpExpclip_by_value*
T0
#
mul_2Mulsplit_1Exp*
T0
2
add/yConst*
dtype0*
valueB
 *??'7
!
addAddmul_2add/y*
T0
?
Sum/reduction_indicesConst*
dtype0*
value	B :
L
SumSumaddSum/reduction_indices*
T0*

Tidx0*
	keep_dims(
4
mul_3/xConst*
dtype0*
valueB
 *  ??
#
mul_3Mulmul_3/xadd*
T0
'
truedivRealDivmul_3Sum*
T0
'
mul_4Mulone_hottruediv*
T0
A
Sum_1/reduction_indicesConst*
dtype0*
value	B :
R
Sum_1Summul_4Sum_1/reduction_indices*
T0*

Tidx0*
	keep_dims( 
4
add_1/yConst*
dtype0*
valueB
 *??'7
%
add_1AddSum_1add_1/y*
T0

LogLogadd_1*
T0
C
mul_5Mulone_hot#Placeholder/old_label_probability_0*
T0
4
add_2/yConst*
dtype0*
valueB
 *??'7
%
add_2Addmul_5add_2/y*
T0
A
Sum_2/reduction_indicesConst*
dtype0*
value	B :
R
Sum_2Sumadd_2Sum_2/reduction_indices*
T0*

Tidx0*
	keep_dims( 

Log_1LogSum_2*
T0
#
add_3AddConst_3Log*
T0
#
sub_4Subadd_3Log_1*
T0

Exp_1Expsub_4*
T0
F
clip_by_value_1/Minimum/yConst*
dtype0*
valueB
 *  @@
M
clip_by_value_1/MinimumMinimumExp_1clip_by_value_1/Minimum/y*
T0
>
clip_by_value_1/yConst*
dtype0*
valueB
 *    
O
clip_by_value_1Maximumclip_by_value_1/Minimumclip_by_value_1/y*
T0
1
mul_6Mulclip_by_value_1	Squeeze_1*
T0
F
clip_by_value_2/Minimum/yConst*
dtype0*
valueB
 *????
M
clip_by_value_2/MinimumMinimumExp_1clip_by_value_2/Minimum/y*
T0
>
clip_by_value_2/yConst*
dtype0*
valueB
 *??L?
O
clip_by_value_2Maximumclip_by_value_2/Minimumclip_by_value_2/y*
T0
1
mul_7Mulclip_by_value_2	Squeeze_1*
T0
)
MinimumMinimummul_6mul_7*
T0
)
mul_8Mul	Squeeze_8Minimum*
T0
5
Const_4Const*
dtype0*
valueB: 
B
Sum_3Summul_8Const_4*
T0*

Tidx0*
	keep_dims( 

Neg_1NegSum_3*
T0
5
Const_5Const*
dtype0*
valueB: 
F
Sum_4Sum	Squeeze_8Const_5*
T0*

Tidx0*
	keep_dims( 
6
	Maximum/yConst*
dtype0*
valueB
 *  ??
-
MaximumMaximumSum_4	Maximum/y*
T0
-
	truediv_1RealDivNeg_1Maximum*
T0
)
add_4AddConst_2	truediv_1*
T0
4
Const_6Const*
dtype0*
valueB
 *    
?
one_hot_1/on_valueConst*
dtype0*
valueB
 *  ??
@
one_hot_1/off_valueConst*
dtype0*
valueB
 *    
9
one_hot_1/depthConst*
dtype0*
value	B :
?
	one_hot_1OneHot	Squeeze_3one_hot_1/depthone_hot_1/on_valueone_hot_1/off_value*
T0*
TI0*
axis?????????
4
sub_5/xConst*
dtype0*
valueB
 *  ??
)
sub_5Subsub_5/x	split_1:1*
T0
4
Pow_2/xConst*
dtype0*
valueB
 *   A
4
Pow_2/yConst*
dtype0*
valueB
 *  ?A
'
Pow_2PowPow_2/xPow_2/y*
T0
#
mul_9Mulsub_5Pow_2*
T0
5
sub_6SubPlaceholder/fc2_label_1mul_9*
T0
A
Max_1/reduction_indicesConst*
dtype0*
value	B :
R
Max_1Maxsub_6Max_1/reduction_indices*
T0*

Tidx0*
	keep_dims(
5
sub_7SubPlaceholder/fc2_label_1Max_1*
T0
4
Pow_3/xConst*
dtype0*
valueB
 *   A
4
Pow_3/yConst*
dtype0*
valueB
 *  ?A
'
Pow_3PowPow_3/xPow_3/y*
T0

Neg_2NegPow_3*
T0
F
clip_by_value_3/Minimum/yConst*
dtype0*
valueB
 *  ??
M
clip_by_value_3/MinimumMinimumsub_7clip_by_value_3/Minimum/y*
T0
C
clip_by_value_3Maximumclip_by_value_3/MinimumNeg_2*
T0
&
Exp_2Expclip_by_value_3*
T0
(
mul_10Mul	split_1:1Exp_2*
T0
4
add_5/yConst*
dtype0*
valueB
 *??'7
&
add_5Addmul_10add_5/y*
T0
A
Sum_5/reduction_indicesConst*
dtype0*
value	B :
R
Sum_5Sumadd_5Sum_5/reduction_indices*
T0*

Tidx0*
	keep_dims(
5
mul_11/xConst*
dtype0*
valueB
 *  ??
'
mul_11Mulmul_11/xadd_5*
T0
,
	truediv_2RealDivmul_11Sum_5*
T0
,
mul_12Mul	one_hot_1	truediv_2*
T0
A
Sum_6/reduction_indicesConst*
dtype0*
value	B :
S
Sum_6Summul_12Sum_6/reduction_indices*
T0*

Tidx0*
	keep_dims( 
4
add_6/yConst*
dtype0*
valueB
 *??'7
%
add_6AddSum_6add_6/y*
T0

Log_2Logadd_6*
T0
F
mul_13Mul	one_hot_1#Placeholder/old_label_probability_1*
T0
4
add_7/yConst*
dtype0*
valueB
 *??'7
&
add_7Addmul_13add_7/y*
T0
A
Sum_7/reduction_indicesConst*
dtype0*
value	B :
R
Sum_7Sumadd_7Sum_7/reduction_indices*
T0*

Tidx0*
	keep_dims( 

Log_3LogSum_7*
T0
%
add_8AddConst_6Log_2*
T0
#
sub_8Subadd_8Log_3*
T0

Exp_3Expsub_8*
T0
F
clip_by_value_4/Minimum/yConst*
dtype0*
valueB
 *  @@
M
clip_by_value_4/MinimumMinimumExp_3clip_by_value_4/Minimum/y*
T0
>
clip_by_value_4/yConst*
dtype0*
valueB
 *    
O
clip_by_value_4Maximumclip_by_value_4/Minimumclip_by_value_4/y*
T0
2
mul_14Mulclip_by_value_4	Squeeze_1*
T0
F
clip_by_value_5/Minimum/yConst*
dtype0*
valueB
 *????
M
clip_by_value_5/MinimumMinimumExp_3clip_by_value_5/Minimum/y*
T0
>
clip_by_value_5/yConst*
dtype0*
valueB
 *??L?
O
clip_by_value_5Maximumclip_by_value_5/Minimumclip_by_value_5/y*
T0
2
mul_15Mulclip_by_value_5	Squeeze_1*
T0
-
	Minimum_1Minimummul_14mul_15*
T0
,
mul_16Mul	Squeeze_9	Minimum_1*
T0
5
Const_7Const*
dtype0*
valueB: 
C
Sum_8Summul_16Const_7*
T0*

Tidx0*
	keep_dims( 

Neg_3NegSum_8*
T0
5
Const_8Const*
dtype0*
valueB: 
F
Sum_9Sum	Squeeze_9Const_8*
T0*

Tidx0*
	keep_dims( 
8
Maximum_1/yConst*
dtype0*
valueB
 *  ??
1
	Maximum_1MaximumSum_9Maximum_1/y*
T0
/
	truediv_3RealDivNeg_3	Maximum_1*
T0
'
add_9Addadd_4	truediv_3*
T0
4
Const_9Const*
dtype0*
valueB
 *    
?
one_hot_2/on_valueConst*
dtype0*
valueB
 *  ??
@
one_hot_2/off_valueConst*
dtype0*
valueB
 *    
9
one_hot_2/depthConst*
dtype0*
value	B :
?
	one_hot_2OneHot	Squeeze_4one_hot_2/depthone_hot_2/on_valueone_hot_2/off_value*
T0*
TI0*
axis?????????
4
sub_9/xConst*
dtype0*
valueB
 *  ??
)
sub_9Subsub_9/x	split_1:2*
T0
4
Pow_4/xConst*
dtype0*
valueB
 *   A
4
Pow_4/yConst*
dtype0*
valueB
 *  ?A
'
Pow_4PowPow_4/xPow_4/y*
T0
$
mul_17Mulsub_9Pow_4*
T0
7
sub_10SubPlaceholder/fc2_label_2mul_17*
T0
A
Max_2/reduction_indicesConst*
dtype0*
value	B :
S
Max_2Maxsub_10Max_2/reduction_indices*
T0*

Tidx0*
	keep_dims(
6
sub_11SubPlaceholder/fc2_label_2Max_2*
T0
4
Pow_5/xConst*
dtype0*
valueB
 *   A
4
Pow_5/yConst*
dtype0*
valueB
 *  ?A
'
Pow_5PowPow_5/xPow_5/y*
T0

Neg_4NegPow_5*
T0
F
clip_by_value_6/Minimum/yConst*
dtype0*
valueB
 *  ??
N
clip_by_value_6/MinimumMinimumsub_11clip_by_value_6/Minimum/y*
T0
C
clip_by_value_6Maximumclip_by_value_6/MinimumNeg_4*
T0
&
Exp_4Expclip_by_value_6*
T0
(
mul_18Mul	split_1:2Exp_4*
T0
5
add_10/yConst*
dtype0*
valueB
 *??'7
(
add_10Addmul_18add_10/y*
T0
B
Sum_10/reduction_indicesConst*
dtype0*
value	B :
U
Sum_10Sumadd_10Sum_10/reduction_indices*
T0*

Tidx0*
	keep_dims(
5
mul_19/xConst*
dtype0*
valueB
 *  ??
(
mul_19Mulmul_19/xadd_10*
T0
-
	truediv_4RealDivmul_19Sum_10*
T0
,
mul_20Mul	one_hot_2	truediv_4*
T0
B
Sum_11/reduction_indicesConst*
dtype0*
value	B :
U
Sum_11Summul_20Sum_11/reduction_indices*
T0*

Tidx0*
	keep_dims( 
5
add_11/yConst*
dtype0*
valueB
 *??'7
(
add_11AddSum_11add_11/y*
T0

Log_4Logadd_11*
T0
F
mul_21Mul	one_hot_2#Placeholder/old_label_probability_2*
T0
5
add_12/yConst*
dtype0*
valueB
 *??'7
(
add_12Addmul_21add_12/y*
T0
B
Sum_12/reduction_indicesConst*
dtype0*
value	B :
U
Sum_12Sumadd_12Sum_12/reduction_indices*
T0*

Tidx0*
	keep_dims( 

Log_5LogSum_12*
T0
&
add_13AddConst_9Log_4*
T0
%
sub_12Subadd_13Log_5*
T0

Exp_5Expsub_12*
T0
F
clip_by_value_7/Minimum/yConst*
dtype0*
valueB
 *  @@
M
clip_by_value_7/MinimumMinimumExp_5clip_by_value_7/Minimum/y*
T0
>
clip_by_value_7/yConst*
dtype0*
valueB
 *    
O
clip_by_value_7Maximumclip_by_value_7/Minimumclip_by_value_7/y*
T0
2
mul_22Mulclip_by_value_7	Squeeze_1*
T0
F
clip_by_value_8/Minimum/yConst*
dtype0*
valueB
 *????
M
clip_by_value_8/MinimumMinimumExp_5clip_by_value_8/Minimum/y*
T0
>
clip_by_value_8/yConst*
dtype0*
valueB
 *??L?
O
clip_by_value_8Maximumclip_by_value_8/Minimumclip_by_value_8/y*
T0
2
mul_23Mulclip_by_value_8	Squeeze_1*
T0
-
	Minimum_2Minimummul_22mul_23*
T0
-
mul_24Mul
Squeeze_10	Minimum_2*
T0
6
Const_10Const*
dtype0*
valueB: 
E
Sum_13Summul_24Const_10*
T0*

Tidx0*
	keep_dims( 

Neg_5NegSum_13*
T0
6
Const_11Const*
dtype0*
valueB: 
I
Sum_14Sum
Squeeze_10Const_11*
T0*

Tidx0*
	keep_dims( 
8
Maximum_2/yConst*
dtype0*
valueB
 *  ??
2
	Maximum_2MaximumSum_14Maximum_2/y*
T0
/
	truediv_5RealDivNeg_5	Maximum_2*
T0
(
add_14Addadd_9	truediv_5*
T0
5
Const_12Const*
dtype0*
valueB
 *    
?
one_hot_3/on_valueConst*
dtype0*
valueB
 *  ??
@
one_hot_3/off_valueConst*
dtype0*
valueB
 *    
9
one_hot_3/depthConst*
dtype0*
value	B :
?
	one_hot_3OneHot	Squeeze_5one_hot_3/depthone_hot_3/on_valueone_hot_3/off_value*
T0*
TI0*
axis?????????
5
sub_13/xConst*
dtype0*
valueB
 *  ??
+
sub_13Subsub_13/x	split_1:3*
T0
4
Pow_6/xConst*
dtype0*
valueB
 *   A
4
Pow_6/yConst*
dtype0*
valueB
 *  ?A
'
Pow_6PowPow_6/xPow_6/y*
T0
%
mul_25Mulsub_13Pow_6*
T0
7
sub_14SubPlaceholder/fc2_label_3mul_25*
T0
A
Max_3/reduction_indicesConst*
dtype0*
value	B :
S
Max_3Maxsub_14Max_3/reduction_indices*
T0*

Tidx0*
	keep_dims(
6
sub_15SubPlaceholder/fc2_label_3Max_3*
T0
4
Pow_7/xConst*
dtype0*
valueB
 *   A
4
Pow_7/yConst*
dtype0*
valueB
 *  ?A
'
Pow_7PowPow_7/xPow_7/y*
T0

Neg_6NegPow_7*
T0
F
clip_by_value_9/Minimum/yConst*
dtype0*
valueB
 *  ??
N
clip_by_value_9/MinimumMinimumsub_15clip_by_value_9/Minimum/y*
T0
C
clip_by_value_9Maximumclip_by_value_9/MinimumNeg_6*
T0
&
Exp_6Expclip_by_value_9*
T0
(
mul_26Mul	split_1:3Exp_6*
T0
5
add_15/yConst*
dtype0*
valueB
 *??'7
(
add_15Addmul_26add_15/y*
T0
B
Sum_15/reduction_indicesConst*
dtype0*
value	B :
U
Sum_15Sumadd_15Sum_15/reduction_indices*
T0*

Tidx0*
	keep_dims(
5
mul_27/xConst*
dtype0*
valueB
 *  ??
(
mul_27Mulmul_27/xadd_15*
T0
-
	truediv_6RealDivmul_27Sum_15*
T0
,
mul_28Mul	one_hot_3	truediv_6*
T0
B
Sum_16/reduction_indicesConst*
dtype0*
value	B :
U
Sum_16Summul_28Sum_16/reduction_indices*
T0*

Tidx0*
	keep_dims( 
5
add_16/yConst*
dtype0*
valueB
 *??'7
(
add_16AddSum_16add_16/y*
T0

Log_6Logadd_16*
T0
F
mul_29Mul	one_hot_3#Placeholder/old_label_probability_3*
T0
5
add_17/yConst*
dtype0*
valueB
 *??'7
(
add_17Addmul_29add_17/y*
T0
B
Sum_17/reduction_indicesConst*
dtype0*
value	B :
U
Sum_17Sumadd_17Sum_17/reduction_indices*
T0*

Tidx0*
	keep_dims( 

Log_7LogSum_17*
T0
'
add_18AddConst_12Log_6*
T0
%
sub_16Subadd_18Log_7*
T0

Exp_7Expsub_16*
T0
G
clip_by_value_10/Minimum/yConst*
dtype0*
valueB
 *  @@
O
clip_by_value_10/MinimumMinimumExp_7clip_by_value_10/Minimum/y*
T0
?
clip_by_value_10/yConst*
dtype0*
valueB
 *    
R
clip_by_value_10Maximumclip_by_value_10/Minimumclip_by_value_10/y*
T0
3
mul_30Mulclip_by_value_10	Squeeze_1*
T0
G
clip_by_value_11/Minimum/yConst*
dtype0*
valueB
 *????
O
clip_by_value_11/MinimumMinimumExp_7clip_by_value_11/Minimum/y*
T0
?
clip_by_value_11/yConst*
dtype0*
valueB
 *??L?
R
clip_by_value_11Maximumclip_by_value_11/Minimumclip_by_value_11/y*
T0
3
mul_31Mulclip_by_value_11	Squeeze_1*
T0
-
	Minimum_3Minimummul_30mul_31*
T0
-
mul_32Mul
Squeeze_11	Minimum_3*
T0
6
Const_13Const*
dtype0*
valueB: 
E
Sum_18Summul_32Const_13*
T0*

Tidx0*
	keep_dims( 

Neg_7NegSum_18*
T0
6
Const_14Const*
dtype0*
valueB: 
I
Sum_19Sum
Squeeze_11Const_14*
T0*

Tidx0*
	keep_dims( 
8
Maximum_3/yConst*
dtype0*
valueB
 *  ??
2
	Maximum_3MaximumSum_19Maximum_3/y*
T0
/
	truediv_7RealDivNeg_7	Maximum_3*
T0
)
add_19Addadd_14	truediv_7*
T0
5
Const_15Const*
dtype0*
valueB
 *    
?
one_hot_4/on_valueConst*
dtype0*
valueB
 *  ??
@
one_hot_4/off_valueConst*
dtype0*
valueB
 *    
9
one_hot_4/depthConst*
dtype0*
value	B :
?
	one_hot_4OneHot	Squeeze_6one_hot_4/depthone_hot_4/on_valueone_hot_4/off_value*
T0*
TI0*
axis?????????
5
sub_17/xConst*
dtype0*
valueB
 *  ??
+
sub_17Subsub_17/x	split_1:4*
T0
4
Pow_8/xConst*
dtype0*
valueB
 *   A
4
Pow_8/yConst*
dtype0*
valueB
 *  ?A
'
Pow_8PowPow_8/xPow_8/y*
T0
%
mul_33Mulsub_17Pow_8*
T0
7
sub_18SubPlaceholder/fc2_label_4mul_33*
T0
A
Max_4/reduction_indicesConst*
dtype0*
value	B :
S
Max_4Maxsub_18Max_4/reduction_indices*
T0*

Tidx0*
	keep_dims(
6
sub_19SubPlaceholder/fc2_label_4Max_4*
T0
4
Pow_9/xConst*
dtype0*
valueB
 *   A
4
Pow_9/yConst*
dtype0*
valueB
 *  ?A
'
Pow_9PowPow_9/xPow_9/y*
T0

Neg_8NegPow_9*
T0
G
clip_by_value_12/Minimum/yConst*
dtype0*
valueB
 *  ??
P
clip_by_value_12/MinimumMinimumsub_19clip_by_value_12/Minimum/y*
T0
E
clip_by_value_12Maximumclip_by_value_12/MinimumNeg_8*
T0
'
Exp_8Expclip_by_value_12*
T0
(
mul_34Mul	split_1:4Exp_8*
T0
5
add_20/yConst*
dtype0*
valueB
 *??'7
(
add_20Addmul_34add_20/y*
T0
B
Sum_20/reduction_indicesConst*
dtype0*
value	B :
U
Sum_20Sumadd_20Sum_20/reduction_indices*
T0*

Tidx0*
	keep_dims(
5
mul_35/xConst*
dtype0*
valueB
 *  ??
(
mul_35Mulmul_35/xadd_20*
T0
-
	truediv_8RealDivmul_35Sum_20*
T0
,
mul_36Mul	one_hot_4	truediv_8*
T0
B
Sum_21/reduction_indicesConst*
dtype0*
value	B :
U
Sum_21Summul_36Sum_21/reduction_indices*
T0*

Tidx0*
	keep_dims( 
5
add_21/yConst*
dtype0*
valueB
 *??'7
(
add_21AddSum_21add_21/y*
T0

Log_8Logadd_21*
T0
F
mul_37Mul	one_hot_4#Placeholder/old_label_probability_4*
T0
5
add_22/yConst*
dtype0*
valueB
 *??'7
(
add_22Addmul_37add_22/y*
T0
B
Sum_22/reduction_indicesConst*
dtype0*
value	B :
U
Sum_22Sumadd_22Sum_22/reduction_indices*
T0*

Tidx0*
	keep_dims( 

Log_9LogSum_22*
T0
'
add_23AddConst_15Log_8*
T0
%
sub_20Subadd_23Log_9*
T0

Exp_9Expsub_20*
T0
G
clip_by_value_13/Minimum/yConst*
dtype0*
valueB
 *  @@
O
clip_by_value_13/MinimumMinimumExp_9clip_by_value_13/Minimum/y*
T0
?
clip_by_value_13/yConst*
dtype0*
valueB
 *    
R
clip_by_value_13Maximumclip_by_value_13/Minimumclip_by_value_13/y*
T0
3
mul_38Mulclip_by_value_13	Squeeze_1*
T0
G
clip_by_value_14/Minimum/yConst*
dtype0*
valueB
 *????
O
clip_by_value_14/MinimumMinimumExp_9clip_by_value_14/Minimum/y*
T0
?
clip_by_value_14/yConst*
dtype0*
valueB
 *??L?
R
clip_by_value_14Maximumclip_by_value_14/Minimumclip_by_value_14/y*
T0
3
mul_39Mulclip_by_value_14	Squeeze_1*
T0
-
	Minimum_4Minimummul_38mul_39*
T0
-
mul_40Mul
Squeeze_12	Minimum_4*
T0
6
Const_16Const*
dtype0*
valueB: 
E
Sum_23Summul_40Const_16*
T0*

Tidx0*
	keep_dims( 

Neg_9NegSum_23*
T0
6
Const_17Const*
dtype0*
valueB: 
I
Sum_24Sum
Squeeze_12Const_17*
T0*

Tidx0*
	keep_dims( 
8
Maximum_4/yConst*
dtype0*
valueB
 *  ??
2
	Maximum_4MaximumSum_24Maximum_4/y*
T0
/
	truediv_9RealDivNeg_9	Maximum_4*
T0
)
add_24Addadd_19	truediv_9*
T0
5
Const_18Const*
dtype0*
valueB
 *    
?
one_hot_5/on_valueConst*
dtype0*
valueB
 *  ??
@
one_hot_5/off_valueConst*
dtype0*
valueB
 *    
9
one_hot_5/depthConst*
dtype0*
value	B :
?
	one_hot_5OneHot	Squeeze_7one_hot_5/depthone_hot_5/on_valueone_hot_5/off_value*
T0*
TI0*
axis?????????
5
sub_21/xConst*
dtype0*
valueB
 *  ??
+
sub_21Subsub_21/x	split_1:5*
T0
5
Pow_10/xConst*
dtype0*
valueB
 *   A
5
Pow_10/yConst*
dtype0*
valueB
 *  ?A
*
Pow_10PowPow_10/xPow_10/y*
T0
&
mul_41Mulsub_21Pow_10*
T0
7
sub_22SubPlaceholder/fc2_label_5mul_41*
T0
A
Max_5/reduction_indicesConst*
dtype0*
value	B :
S
Max_5Maxsub_22Max_5/reduction_indices*
T0*

Tidx0*
	keep_dims(
6
sub_23SubPlaceholder/fc2_label_5Max_5*
T0
5
Pow_11/xConst*
dtype0*
valueB
 *   A
5
Pow_11/yConst*
dtype0*
valueB
 *  ?A
*
Pow_11PowPow_11/xPow_11/y*
T0

Neg_10NegPow_11*
T0
G
clip_by_value_15/Minimum/yConst*
dtype0*
valueB
 *  ??
P
clip_by_value_15/MinimumMinimumsub_23clip_by_value_15/Minimum/y*
T0
F
clip_by_value_15Maximumclip_by_value_15/MinimumNeg_10*
T0
(
Exp_10Expclip_by_value_15*
T0
)
mul_42Mul	split_1:5Exp_10*
T0
5
add_25/yConst*
dtype0*
valueB
 *??'7
(
add_25Addmul_42add_25/y*
T0
B
Sum_25/reduction_indicesConst*
dtype0*
value	B :
U
Sum_25Sumadd_25Sum_25/reduction_indices*
T0*

Tidx0*
	keep_dims(
5
mul_43/xConst*
dtype0*
valueB
 *  ??
(
mul_43Mulmul_43/xadd_25*
T0
.

truediv_10RealDivmul_43Sum_25*
T0
-
mul_44Mul	one_hot_5
truediv_10*
T0
B
Sum_26/reduction_indicesConst*
dtype0*
value	B :
U
Sum_26Summul_44Sum_26/reduction_indices*
T0*

Tidx0*
	keep_dims( 
5
add_26/yConst*
dtype0*
valueB
 *??'7
(
add_26AddSum_26add_26/y*
T0

Log_10Logadd_26*
T0
F
mul_45Mul	one_hot_5#Placeholder/old_label_probability_5*
T0
5
add_27/yConst*
dtype0*
valueB
 *??'7
(
add_27Addmul_45add_27/y*
T0
B
Sum_27/reduction_indicesConst*
dtype0*
value	B :
U
Sum_27Sumadd_27Sum_27/reduction_indices*
T0*

Tidx0*
	keep_dims( 

Log_11LogSum_27*
T0
(
add_28AddConst_18Log_10*
T0
&
sub_24Subadd_28Log_11*
T0

Exp_11Expsub_24*
T0
G
clip_by_value_16/Minimum/yConst*
dtype0*
valueB
 *  @@
P
clip_by_value_16/MinimumMinimumExp_11clip_by_value_16/Minimum/y*
T0
?
clip_by_value_16/yConst*
dtype0*
valueB
 *    
R
clip_by_value_16Maximumclip_by_value_16/Minimumclip_by_value_16/y*
T0
3
mul_46Mulclip_by_value_16	Squeeze_1*
T0
G
clip_by_value_17/Minimum/yConst*
dtype0*
valueB
 *????
P
clip_by_value_17/MinimumMinimumExp_11clip_by_value_17/Minimum/y*
T0
?
clip_by_value_17/yConst*
dtype0*
valueB
 *??L?
R
clip_by_value_17Maximumclip_by_value_17/Minimumclip_by_value_17/y*
T0
3
mul_47Mulclip_by_value_17	Squeeze_1*
T0
-
	Minimum_5Minimummul_46mul_47*
T0
-
mul_48Mul
Squeeze_13	Minimum_5*
T0
6
Const_19Const*
dtype0*
valueB: 
E
Sum_28Summul_48Const_19*
T0*

Tidx0*
	keep_dims( 

Neg_11NegSum_28*
T0
6
Const_20Const*
dtype0*
valueB: 
I
Sum_29Sum
Squeeze_13Const_20*
T0*

Tidx0*
	keep_dims( 
8
Maximum_5/yConst*
dtype0*
valueB
 *  ??
2
	Maximum_5MaximumSum_29Maximum_5/y*
T0
1

truediv_11RealDivNeg_11	Maximum_5*
T0
*
add_29Addadd_24
truediv_11*
T0
(
mul_49Multruedivsplit_1*
T0
5
add_30/yConst*
dtype0*
valueB
 *??'7
)
add_30Addtruedivadd_30/y*
T0

Log_12Logadd_30*
T0
&
mul_50Mulmul_49Log_12*
T0
B
Sum_30/reduction_indicesConst*
dtype0*
value	B :
U
Sum_30Summul_50Sum_30/reduction_indices*
T0*

Tidx0*
	keep_dims( 

Neg_12NegSum_30*
T0
)
mul_51MulNeg_12	Squeeze_8*
T0
6
Const_21Const*
dtype0*
valueB: 
E
Sum_31Summul_51Const_21*
T0*

Tidx0*
	keep_dims( 

Neg_13NegSum_31*
T0
6
Const_22Const*
dtype0*
valueB: 
H
Sum_32Sum	Squeeze_8Const_22*
T0*

Tidx0*
	keep_dims( 
8
Maximum_6/yConst*
dtype0*
valueB
 *  ??
2
	Maximum_6MaximumSum_32Maximum_6/y*
T0
1

truediv_12RealDivNeg_13	Maximum_6*
T0
,
mul_52Mul	truediv_2	split_1:1*
T0
5
add_31/yConst*
dtype0*
valueB
 *??'7
+
add_31Add	truediv_2add_31/y*
T0

Log_13Logadd_31*
T0
&
mul_53Mulmul_52Log_13*
T0
B
Sum_33/reduction_indicesConst*
dtype0*
value	B :
U
Sum_33Summul_53Sum_33/reduction_indices*
T0*

Tidx0*
	keep_dims( 

Neg_14NegSum_33*
T0
)
mul_54MulNeg_14	Squeeze_9*
T0
6
Const_23Const*
dtype0*
valueB: 
E
Sum_34Summul_54Const_23*
T0*

Tidx0*
	keep_dims( 

Neg_15NegSum_34*
T0
6
Const_24Const*
dtype0*
valueB: 
H
Sum_35Sum	Squeeze_9Const_24*
T0*

Tidx0*
	keep_dims( 
8
Maximum_7/yConst*
dtype0*
valueB
 *  ??
2
	Maximum_7MaximumSum_35Maximum_7/y*
T0
1

truediv_13RealDivNeg_15	Maximum_7*
T0
,
mul_55Mul	truediv_4	split_1:2*
T0
5
add_32/yConst*
dtype0*
valueB
 *??'7
+
add_32Add	truediv_4add_32/y*
T0

Log_14Logadd_32*
T0
&
mul_56Mulmul_55Log_14*
T0
B
Sum_36/reduction_indicesConst*
dtype0*
value	B :
U
Sum_36Summul_56Sum_36/reduction_indices*
T0*

Tidx0*
	keep_dims( 

Neg_16NegSum_36*
T0
*
mul_57MulNeg_16
Squeeze_10*
T0
6
Const_25Const*
dtype0*
valueB: 
E
Sum_37Summul_57Const_25*
T0*

Tidx0*
	keep_dims( 

Neg_17NegSum_37*
T0
6
Const_26Const*
dtype0*
valueB: 
I
Sum_38Sum
Squeeze_10Const_26*
T0*

Tidx0*
	keep_dims( 
8
Maximum_8/yConst*
dtype0*
valueB
 *  ??
2
	Maximum_8MaximumSum_38Maximum_8/y*
T0
1

truediv_14RealDivNeg_17	Maximum_8*
T0
,
mul_58Mul	truediv_6	split_1:3*
T0
5
add_33/yConst*
dtype0*
valueB
 *??'7
+
add_33Add	truediv_6add_33/y*
T0

Log_15Logadd_33*
T0
&
mul_59Mulmul_58Log_15*
T0
B
Sum_39/reduction_indicesConst*
dtype0*
value	B :
U
Sum_39Summul_59Sum_39/reduction_indices*
T0*

Tidx0*
	keep_dims( 

Neg_18NegSum_39*
T0
*
mul_60MulNeg_18
Squeeze_11*
T0
6
Const_27Const*
dtype0*
valueB: 
E
Sum_40Summul_60Const_27*
T0*

Tidx0*
	keep_dims( 

Neg_19NegSum_40*
T0
6
Const_28Const*
dtype0*
valueB: 
I
Sum_41Sum
Squeeze_11Const_28*
T0*

Tidx0*
	keep_dims( 
8
Maximum_9/yConst*
dtype0*
valueB
 *  ??
2
	Maximum_9MaximumSum_41Maximum_9/y*
T0
1

truediv_15RealDivNeg_19	Maximum_9*
T0
,
mul_61Mul	truediv_8	split_1:4*
T0
5
add_34/yConst*
dtype0*
valueB
 *??'7
+
add_34Add	truediv_8add_34/y*
T0

Log_16Logadd_34*
T0
&
mul_62Mulmul_61Log_16*
T0
B
Sum_42/reduction_indicesConst*
dtype0*
value	B :
U
Sum_42Summul_62Sum_42/reduction_indices*
T0*

Tidx0*
	keep_dims( 

Neg_20NegSum_42*
T0
*
mul_63MulNeg_20
Squeeze_12*
T0
6
Const_29Const*
dtype0*
valueB: 
E
Sum_43Summul_63Const_29*
T0*

Tidx0*
	keep_dims( 

Neg_21NegSum_43*
T0
6
Const_30Const*
dtype0*
valueB: 
I
Sum_44Sum
Squeeze_12Const_30*
T0*

Tidx0*
	keep_dims( 
9
Maximum_10/yConst*
dtype0*
valueB
 *  ??
4

Maximum_10MaximumSum_44Maximum_10/y*
T0
2

truediv_16RealDivNeg_21
Maximum_10*
T0
-
mul_64Mul
truediv_10	split_1:5*
T0
5
add_35/yConst*
dtype0*
valueB
 *??'7
,
add_35Add
truediv_10add_35/y*
T0

Log_17Logadd_35*
T0
&
mul_65Mulmul_64Log_17*
T0
B
Sum_45/reduction_indicesConst*
dtype0*
value	B :
U
Sum_45Summul_65Sum_45/reduction_indices*
T0*

Tidx0*
	keep_dims( 

Neg_22NegSum_45*
T0
*
mul_66MulNeg_22
Squeeze_13*
T0
6
Const_31Const*
dtype0*
valueB: 
E
Sum_46Summul_66Const_31*
T0*

Tidx0*
	keep_dims( 

Neg_23NegSum_46*
T0
6
Const_32Const*
dtype0*
valueB: 
I
Sum_47Sum
Squeeze_13Const_32*
T0*

Tidx0*
	keep_dims( 
9
Maximum_11/yConst*
dtype0*
valueB
 *  ??
4

Maximum_11MaximumSum_47Maximum_11/y*
T0
2

truediv_17RealDivNeg_23
Maximum_11*
T0
5
Const_33Const*
dtype0*
valueB
 *    
,
add_36AddConst_33
truediv_12*
T0
*
add_37Addadd_36
truediv_13*
T0
*
add_38Addadd_37
truediv_14*
T0
*
add_39Addadd_38
truediv_15*
T0
*
add_40Addadd_39
truediv_16*
T0
*
add_41Addadd_40
truediv_17*
T0
#
add_42Addmuladd_29*
T0
5
mul_67/xConst*
dtype0*
valueB
 *???<
(
mul_67Mulmul_67/xadd_41*
T0
&
add_43Addadd_42mul_67*
T0
0
exp_output/IdentityIdentityadd_43*
T0 