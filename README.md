# Neural-Network-for-classifing-iris-in-Google-spread-sheet
[The google sheet](https://docs.google.com/spreadsheets/d/1kZih1N1NpMMOKKmTtfIX1BIpiqwvwt-7wuG1b05WFpc/edit?usp=sharing)

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
- **M1** = **X** @ **W1**
- **S1** = **M1** + **B1**
- **A1** = σ(**S1**)
- **M2** = **A1** @ **W2**
- **S2** = **M2** + **B2**
- **A2** = softmax(**S2**)
- **Y^** = **A2**
- L(**Y**, **Y^**) = (**Y**, **Y^**) ** 2
### Expanded expression
- **M1**[i,j] = **Σ**(k=0;InputCount) { **X**[i,k] * **W1**[k,j] }
- **S1**[i,j] = **M1**[i,j] + **B1**[j]
- **A1**[i,j] = σ(**S1**[i,j])
- **M2**[i,j] = **Σ**(k=0;HiddenCount) { **A1**[i,k] * **W2**[k,j] }
- **S2**[i,j] = **M2**[i,j] + **B2**[j]
- **A2**[i,j] = σ(**S1**[i,j]) / (**Σ**(k=0;OutputCount) { σ(**S2**[i,k]) })
- **Y^**[i,j] = **A2**[i,j]
- L(**Y**, Y^)[i] = **Σ**(k=0;OutputCount) { (**Y**[i,k] - **Y^**[i,k]) ** 2 }
