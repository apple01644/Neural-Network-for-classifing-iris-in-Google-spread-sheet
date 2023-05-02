# Neural-Network-for-classifing-iris-in-Google-spread-sheet
[The google sheet](https://docs.google.com/spreadsheets/d/1kZih1N1NpMMOKKmTtfIX1BIpiqwvwt-7wuG1b05WFpc/edit?usp=sharing)

## Result
![image](https://user-images.githubusercontent.com/36323843/235649390-3f63d2bd-e968-4d4d-8c39-dad8893e7e08.png)  
It seems like it can distinguish `setosa`(j=1) and others but `versicolor`(j=2) and `virginica`(j=3).  
I tried to add a hidden layer one more. 
But I still can not. And it was too laggy to train further.

## Matrices and constants
- Constant
  - InputCount = 4
     sepal length (cm), sepal width (cm), petal length (cm), petal width (cm).
  - OutputCount = 3
     setosa, versicolor, or viginica.
  - BatchSize = 20
  - LearningRate = 0.5
- Matrices
  -  **X** ∈ R[BatchSize × InputCount]
     `X is input data`
  - **Y** ∈ R[BatchSize × OutputCount]
     `Y is output data`
  - **Y^** ∈ R[BatchSize × OutputCount]
     `Y^ is predicted output data`
  - **W1** ∈ R[InputCount × HiddenCount]
  - **B1** ∈ R[HiddenCount]
  - **W2** ∈ R[HiddenCount × OutputCount]
  - **B2** ∈ R[OutputCount]
  - some matrices for forward procedure(M1, S1, A2, S1... or etc.)  


![image](https://user-images.githubusercontent.com/36323843/235649033-1ab04407-8980-49db-9b11-adf6fc6fa7a9.png)
## Procedures
1. Initialize weights and biases with random values between `-1` and `+1`
2. Run **Forward**, a procedure to calculate **Y^**
3. Caculate **Loss**, (How far the predicted values from dataset)
4. Run **Backpropagation**,  a procedure to get partial of **Loss** with respect to matrices for forward procedure.
5. Assign new weights and biases.
    **W1**(new)`i,j` = **W1**(old)`i,j` - `learning_rate` * ∂(**Loss**)/∂(**W1**(old)`i,j`
6. Go to `2.`
###  Forward
#### Simple expression
- Step 1: **M1** = **X** @ **W1**
- Step 2: **S1** = **M1** + **B1**
- Step 3: **A1** = σ(**S1**)
- Step 4: **M2** = **A1** @ **W2**
- Step 5: **S2** = **M2** + **B2**
- Step 6: **A2** = softmax(**S2**)
- Step 7: **Y^** = **A2**

### Expanded expression
- Step 1: **M1**[i,j] = **Σ**(k=0;InputCount) { **X**[i,k] * **W1**[k,j] }
- Step 2: **S1**[i,j] = **M1**[i,j] + **B1**[j]
- Step 3: **A1**[i,j] = σ(**S1**[i,j])
- Step 4: **M2**[i,j] = **Σ**(k=0;HiddenCount) { **A1**[i,k] * **W2**[k,j] }
- Step 5: **S2**[i,j] = **M2**[i,j] + **B2**[j]
- Step 6: **A2**[i,j] = σ(**S2**[i,j]) / (**Σ**(k=0;OutputCount) { σ(**S2**[i,k]) })
- Step 7: **Y^**[i,j] = **A2**[i,j]

### Caculate Loss
- Simple: L(**Y**[i], **Y^**[i]) = (**Y**[i] - **Y^**[i]) ** 2
- Expanded: L(**Y**[i], **Y^**[i]) = **Σ**(k=0;OutputCount) { (**Y**[i,k] - **Y^**[i,k]) ** 2 }  


![image](https://user-images.githubusercontent.com/36323843/235649088-7c50e8a0-a508-42fd-bda8-0a7abccf2b7c.png)

### Backpropagation
#### Partial derivative step by step
- Forward
  - Step1:
    - ∂(**M1**[i,j])/∂(**W1**[j,k]) = **A1**[i,k]
  - Step2:
    - ∂(**S1**[i,j])/∂(**M1**[i,j]) = 1
    - ∂(**S1**[i,j])/∂(**B1**[i,j]) = 1
  - Step3: ∂(**A1**[i,j])/∂(**S1**[i,j]) = σ(**S1**[i,j]) * σ(1 - **S1**[i,j])
  - Step4:
    - ∂(**M2**[i,j])/∂(**W2**[j,k]) = **A1**[i,k]
    - ∂(**M2**[i,j])/∂(**A1**[i,k) = **W2**[k,j]
  - Step5: 
    - ∂(**S2**[i,j])/∂(**M2**[i,j]) = 1
    - ∂(**S2**[i,j])/∂(**B2**[i,j]) = 1
  - Step6:  ∂(**A2**[i,j])/∂(**S2**[i,j]) = σ(**S2**[i,j]) * σ(1 - **S2**[i,j])
  - Step7: ∂(**Y^**[i,j])/∂(**A2**[i,j]) = 1
- Caculate Loss: ∂L(**Y**[i], **Y^**[i])/∂(**Y^**[i,j]) = 2×**Y**[i,j] + 2×**Y^**[i,j]  


![image](https://user-images.githubusercontent.com/36323843/235649244-7b030892-97a0-421e-bc0c-0cfe05599850.png)

#### Use the chainrule to caculate partital of Loss(=**L**) to respect to every weights and bias using earlier results of partial derivative
 - Example: ∂(**Loss**[i])/∂(**S2**[i,j]) = (2×**Y**[i,j] + 2×**Y^**[i,j]) * 1 * σ(**S2**[i,j]) * σ(1 - **S2**[i,j])  


 ![image](https://user-images.githubusercontent.com/36323843/235649199-ff18b574-0da2-4bad-bf1a-e59203b3023d.png)


#### Assign new weights and biases. And go to "2." and repeat.
![image](https://user-images.githubusercontent.com/36323843/235649142-7c42177e-9122-4bb7-92d9-1cf3a1d4c0b0.png)

### Train Data
![image](https://user-images.githubusercontent.com/36323843/235648949-a241a3f4-2336-438b-bb95-b984f29be3cf.png)
