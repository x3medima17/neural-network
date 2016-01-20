#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <ctime>
#include <cstdlib>

using namespace std;




class Matrix
{
public:
    int n,m;
    vector<vector<double> > data;

    Matrix(){}

    Matrix(int n,int m)
    {
        this->n = n;
        this->m = m;
        data.resize(n,vector<double>(m,0));
    }


    void set_all(double val)
    {
        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++)
                data[i][j] = val;
    }

    vector<int> size()
    {
        vector<int> tmp(2);
        tmp[0] = n;
        tmp[1] = m;
        return tmp;
    }

    void set(int i,int j, double val)
    {
        this->data[i][j] = val;
    }

    void load_from_file(string src)
    {
        ifstream fin(src.c_str());
        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++)
                fin>>data[i][j];
    }

    Matrix transpose()
    {
        Matrix tmp(m,n);
        tmp.n = m;
        tmp.m = n;
        for(int i=0;i<m;i++)
            for(int j=0;j<n;j++)
                tmp.data[i][j] = data[j][i];
        return tmp;
    }

    Matrix vectorize()
    {
        Matrix tmp(n*m,1);

        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++)
                tmp.data[i*m+j][0] = this->data[i][j];
        return tmp;
    }

    Matrix reshape(int rows,int cols)
    {
        Matrix tmp(rows,cols);
        for(int i=0;i<rows;i++)
            for(int j=0;j<cols;j++)
                tmp.data[i][j] = data[i*cols+j][0];
        return tmp;
    }

    double maxx()
    {
        int maxx = -999999;
        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++)
                if(data[i][j]>maxx)
                    maxx = data[i][j];
        return maxx;
    }

    double sum()
    {
        double s = 0;
        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++)
                s += data[i][j];
        return s;
    }

    Matrix slice(int a,int b, int c, int d)
    {
        if(a == -1)
            a = 1;
        if(b == -1)
            b = n;
        if(c == -1)
            c = 1;
        if(d == -1)
            d = m;

        Matrix tmp(b-a+1,d-c+1);
        int col=0,row=0;
        for(int i=a;i<=b;i++)
        {
            col = 0;
            for(int j=c;j<=d;j++)
            {
                tmp.data[row][col] = data[i-1][j-1];
                col++;
            }
            row++;
        }
        return tmp;
    }

    void randomize()
    {
        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++)
                data[i][j] = ((double) rand() / (RAND_MAX)) ;
    }


};

ostream& operator<<(ostream& os, const vector<int> &V)
{
    cout<<V[0]<<" "<<V[1]<<endl;
}

ostream& operator<<(ostream& os, const vector<vector<double> > &V)
{
    int n = V.size();
    int m = V[0].size();

    for(int i=0;i<n;i++)
    {
        for(int j=0;j<m;j++)
            cout<<V[i][j]<<' ';
        cout<<endl;
    }

}

ostream& operator<<(ostream& os, const Matrix  &M)
{
    int n = M.n;
    int m = M.m;
    //cout<<n<<" "<<m<<endl;
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<m;j++)
            cout<<M.data[i][j]<<' ';
        cout<<endl;
    }
    return os;
}


Matrix operator+(const Matrix A,  const Matrix B )
{
    int n = (&A)->n;
    int m = (&A)->m;
    Matrix curr(n,m);
    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            curr.data[i][j] = A.data[i][j] + B.data[i][j];
    return curr;
}

Matrix operator-(const Matrix A,  const Matrix B )
{
    int n = (&A)->n;
    int m = (&A)->m;
    Matrix curr(n,m);
    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            curr.data[i][j] = A.data[i][j] - B.data[i][j];
    return curr;
}

template<typename T>
Matrix operator+(const Matrix A, const T val)
{
    int n = A.n;
    int m = A.m;
    Matrix tmp(n,m);

    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            tmp.data[i][j] = A.data[i][j] + val;
    return tmp;
}


Matrix operator*(const Matrix A, const Matrix B)
{
    int n = A.n;
    int m = B.m;
    assert(A.m == B.n);
    Matrix curr(n,m);

    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            for(int k=0;k<A.m;k++)
                curr.data[i][j] += A.data[i][k] * B.data[k][j];

    return curr;


}

Matrix operator*=(const Matrix A, const Matrix B)
{
    int n = A.n;
    int m = A.m;

    Matrix curr(n,m);

    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            curr.data[i][j] = A.data[i][j] * B.data[i][j];

    return curr;
}

template<typename T>
Matrix operator*(const Matrix A, const T val)
{
    int n = A.n;
    int m = A.m;
    Matrix tmp(n,m);

    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            tmp.data[i][j] = A.data[i][j] * val;
    return tmp;
}

Matrix operator||(const Matrix A, const Matrix B)
{
    int an = A.n;
    int am = A.m;
    int bn = B.n;
    int bm = B.m;
    assert(an == bn);
    Matrix tmp(an,am+bm);
    for(int i=0;i<an;i++)
        for(int j=0;j<am+bm;j++)
            {
                if(j>= am)
                    tmp.data[i][j+am-1] = B.data[i][j-am];
                else
                    tmp.data[i][j] = A.data[i][j];
            }
    return tmp;
}



struct cost
{
    double J;
    Matrix Theta1_grad;
    Matrix Theta2_grad;
};

Matrix log(Matrix M)
{
    int n = M.size()[0];
    int m = M.size()[1];
    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            M.set(i,j, log(M.data[i][j]));
    return M;
}

inline double sigmoid(double z)
{
    return 1.0 / (1.0 + exp(-z));
}

Matrix sigmoid(Matrix M)
{
    int n = M.size()[0];
    int m = M.size()[1];
    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            M.set(i,j, sigmoid(M.data[i][j]));
    return M;

}

Matrix sigmoidGradient(Matrix M)
{
    int n = M.size()[0];
    int m = M.size()[1];
    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            M.set(i,j,sigmoid(M.data[i][j]) * (1-sigmoid(M.data[i][j])));
    return M;

}


Matrix vectorize_y(Matrix &y)
{
    int m = y.size()[0];
    int num_labels = y.maxx()+1;
    Matrix Y(m,num_labels);
    for(int i=0;i<m;i++)
        Y.set(i,y.data[i][0],1);
    return Y;
}

cost CostFunction(int input_layer_size, int hidden_layer_size, int num_labels,
                    Matrix &Theta1, Matrix &Theta2,Matrix &X, Matrix &y)
{
    Matrix Y = vectorize_y(y);
    cost curr_cost;
    curr_cost.J = 0;
    int m = X.size()[0];
    Matrix ones(m,1);
    ones.set_all(1);

    //Feedforward propagation
    Matrix a1 = ones||X;

    Matrix z2 = a1 * Theta1.transpose();
    Matrix a2 = ones||sigmoid(z2);

    Matrix z3 = a2 * Theta2.transpose();
    Matrix a3 = sigmoid(z3);
    Matrix h0 = a3;

    Matrix p1 = (Y * (-1)) *= log(h0);
    Matrix p2 = (Y* (-1) +1) *= log(h0 * (-1) +1);

    Matrix a = (p1-p2);
    curr_cost.J = a.sum() * (1.0/m);

    //Gradient
    Matrix delta3 = a3 - Y;

    Matrix delta2 = (Theta2.transpose()  * delta3.transpose()).transpose();
    delta2 = delta2.slice(-1,-1,2,-1) *= sigmoidGradient(z2);

    Matrix Delta1 = delta2.transpose() * a1;
    Matrix Delta2 = delta3.transpose() * a2;

    double coeff = 1.0 / m;

    Matrix D1 = Delta1 * coeff;
    Matrix D2 = Delta2 * coeff;

    curr_cost.Theta1_grad = D1;
    curr_cost.Theta2_grad = D2;

    return curr_cost;

}

int predict(Matrix &Theta1, Matrix &Theta2, Matrix &X)
{
    Matrix ones(1,1);
    ones.set_all(1);

    Matrix h1 = sigmoid((ones||X) * Theta1.transpose());
    Matrix h2 = sigmoid((ones||h1) * Theta2.transpose());
    double maxx=0;
    int maxi = 0;
    for(int i=0;i<h2.size()[1];i++)
        if(maxx<h2.data[0][i])
        {
            maxx = h2.data[0][i];
            maxi = i;
        }
    cout<<endl<<h2.transpose();
    return maxi;
}

int main()
{
    Matrix data(30,42 );
    data.load_from_file("NN.dat");

    // Load data
    Matrix X = data.slice(-1,-1,-1,41);
    Matrix y = data.slice(-1,-1,42,42);

    //Set variables
    int input_layer_size = X.size()[1];
    int hidden_layer_size = input_layer_size*3/2;
    int num_labels = y.maxx()+1;
    int m = X.size()[0];
    double coeff = 1.0 / m;
    double alpha = -71;
    int iters = 30;

    //Set Theta
    Matrix Theta1(hidden_layer_size,input_layer_size+1);
    Matrix Theta2(num_labels,hidden_layer_size+1);

    cost curr_cost;
    Theta1.randomize();
    Theta2.randomize();

    for(int i=0;i<iters;i++)
    {
        curr_cost = CostFunction(input_layer_size,hidden_layer_size,num_labels,
                                Theta1,Theta2, X, y );
        Theta1 = Theta1 + (curr_cost.Theta1_grad * (coeff*alpha));
        Theta2 = Theta2 + (curr_cost.Theta2_grad * (coeff*alpha));
        cout<<curr_cost.J<<endl;

    }
    Matrix x = X.slice(5,5,-1,-1);
    cout<<predict(Theta1,Theta2,x);

    return 0;
}
