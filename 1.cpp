#include <iostream>
#include <vector> 
#include <cmath>

using namespace std;

vector<double> round_vector(vector<double> Pom){
    for(int i=0;i<3;i++)
        Pom[i] = round(Pom[i]);
    return Pom;
}
vector<double> saberi(vector<double> P1, vector<double> P2, vector<double> P3){
    vector<double> Pom(3);
    for(int i=0;i<3;i++)
        Pom[i] = P1[i] + P2[i] + P3[i];
    return Pom;
}
vector<double> podeli(vector<double> Pom){
    for(int i=0;i<3;i++)
        Pom[i] /= 3.0;
    return Pom;
}
vector<double> cross(vector<double> P, vector<double> Q){
    vector<double> Res(3);

    Res[0] = P[1]*Q[2] - P[2]*Q[1];
    Res[1] = P[2]*Q[0] - P[0]*Q[2];
    Res[2] = P[0]*Q[1] - P[1]*Q[0];
    return Res;
}
vector<double> afina(vector<double> T){
    vector<double> Pom(3);
    for(int i=0;i<3;i++)
        Pom[i] = T[i]/T[2];
    return Pom;
}
vector<double> projektivna(vector<double> T){
    vector<double> Pom(3);
    Pom[0] = T[0];
    Pom[1] = T[1];
    Pom[2] = 1.0;
    return Pom;
}
vector<double> Nevidljivo(vector<double> T1, vector<double> T2, vector<double> T3, vector<double> T5, vector<double> T6, vector<double> T7, vector<double> T8){
    T1 = projektivna(T1);
    T2 = projektivna(T2);
    T3 = projektivna(T3);
    T5 = projektivna(T5);
    T6 = projektivna(T6);
    T7 = projektivna(T7);
    T8 = projektivna(T8);
    vector<double> X1inf = round_vector(afina(cross(cross(T2, T6), cross(T1, T5))));
    vector<double> X2inf = round_vector(afina(cross(cross(T2, T6), cross(T3, T7))));
    vector<double> X3inf = round_vector(afina(cross(cross(T1, T5), cross(T3, T7))));
    vector<double> Xinf = round_vector(podeli(saberi(X1inf, X2inf, X3inf)));

    vector<double> Y1inf = round_vector(afina(cross(cross(T5, T6), cross(T7, T8))));
    vector<double> Y2inf = round_vector(afina(cross(cross(T1, T2), cross(T7, T8))));
    vector<double> Y3inf = round_vector(afina(cross(cross(T1, T2), cross(T5, T6))));
    vector<double> Yinf = round_vector(podeli(saberi(Y1inf, Y2inf, Y3inf)));

    return round_vector(afina(cross(cross(Xinf, T8), cross(Yinf, T3))));
}
int main(){

    //Koordinate sa mog primera
    vector<double> T1 = {520, 247};
    vector<double> T2 = {355, 435};
    vector<double> T3 = {172, 345};
    vector<double> T5 = {525, 147};
    vector<double> T6 = {345, 311};
    vector<double> T7 = {142, 236};
    vector<double> T8 = {360, 121};


    /*vector<double> T1 = {595, 301};
    vector<double> T2 = {292, 517};
    vector<double> T3 = {157, 379};
    vector<double> T5 = {665, 116};
    vector<double> T6 = {304, 295};
    vector<double> T7 = {135, 163};
    vector<double> T8 = {509, 43};
    */


    vector<double> T4 = Nevidljivo(T1, T2, T3, T5, T6, T7, T8);
    cout << "T4 = {" << T4[0] << ", " << T4[1] << ", " << T4[2];
    cout << "}" << endl;
    return 0;
}