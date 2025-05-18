% hosting capacity maximization (kite mesh)
clc; clear all

% power injections (negative load)
S=-[0.224    0.708    1.572    0.072]*exp(j*.3176); % acos(0.95) ind
I=conj(S).';

% line admittances of the kite grid 
y12=1-j*10;
y13=2*y12;
y23=3-j*20;
y34=y23;
y45=2*y12;

% kite admittance matrix (slack on 5)
Y=[y12+y13 -y12     -y13          0   0
  -y12      y12+y23 -y23          0   0
  -y13     -y23      y13+y23+y34 -y34 0
   0        0       -y34          y34+y45  -y45];

% AR rnd process for injections
i1w(1)=real(I(1)); % in node 1
m=500; rng(98);
e4=randn(4,m)*.25;
e1=randn(1,m)*.5;
for t=1:m-1
    I(:,t+1)=0.65*I(:,t)+e4(:,t);  % all other nodes
    i1w(t+1)=0.75*i1w(t)+e1(t);
    I(1,t+1)=-i1w(t+1)+j*imag(I(1,t+1)); % (-)i1w from active load to inj.

    v=1+Y(1:4,1:4)^-1*I(:,t+1); % ohm & kirchhoff's circuit laws
    I12=y12*(v(1)-v(2)); i12(t+1)=abs(I12)*sign(real(I12)); 
    I13=y13*(v(1)-v(3)); i13(t+1)=abs(I13)*sign(real(I13));
    I23=y23*(v(2)-v(3)); i23(t+1)=abs(I23)*sign(real(I23));
end
Ii=abs(I).*sign(real(I));  % abs & sign to repliclate meter RMS info

title('input (plot 1) / output (plot 2)') 
subplot(2,1,1),  plot(1:m,I)
xlabel('Time stamp')
ylabel('Nodal injections') 

subplot(2,1,2),  plot(1:m,([i12; i13; i23]))
xlabel('Time stamp')
ylabel('Line currents') 

rate_12=inf; %.25; 
rate_13=1; 
rate_23=inf; %.50
ij=[i12' i13' i23'];
Ii=Ii';
k=0;
for t=1:m
    pos_vio=max([ ij(t,1)-rate_12  ij(t,2)-rate_13  ij(t,3)-rate_23]);
    neg_vio=max([-ij(t,1)-rate_12 -ij(t,2)-rate_13 -ij(t,3)-rate_23]);
    if  pos_vio > 0
        k=k+1; y(k,:)=pos_vio; X(k,1:4)=Ii(t,1:4);
    end
%     if  neg_vio > 0 
%         k=k+1; y(k,:)=neg_vio; X(k,1:4)=-Ii(t,1:4);
%     end
end

X=[X ones(k,1)];
beta =(X'*X)^-1*X'*y

res=y-X*beta;
norm(res)/k