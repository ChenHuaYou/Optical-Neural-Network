clear;
close all;
N=128;  
% mask1=exp(i*rand(N,N)*2*pi);
load 'c:\mask1.mat';
cmask1=conj(mask1);
% save ('c:\mask1.mat', 'mask1') ;

chemin='c:\imgsWan\Train\';%ͼƬ·��
Image=dir('c:\imgsWan\Train\');
Number=length(Image)-2;
for nn=1:(Number)

		XRGB=imread(strcat(chemin,num2str(nn),'.jpg'));
		X0=rgb2gray(XRGB);                       %��ɫͼ��ת��Ϊ�Ҷ�ͼ��
		% X0=(XRGB);  
		X0=imresize(X0,[128 128],'bicubic');
		% X0=(XRGB);
		[M0,N0]=size(X0);                            %��ȡ�Ҷ�ͼ�����������С
		N1=min(M0,N0);
		%��Ϣͼȡ����, �ɰ���Ҫ�޸�
		m0=1;                                          %ͼ�����ؽ������е���ʾ����, 
		X1=imresize(X0,N/N1*m0);
		[M1,N1]=size(X1);
		X=zeros(N,N);
		X(N/2-M1/2+1:N/2+M1/2,N/2-N1/2+1:N/2+N1/2)=X1(1:M1,1:N1);

		h=0.532e-3;      %����(mm), �ɰ���Ҫ�޸�
		w=h;
		k=2*pi/h;
		pix=0.0064;      %SLM���ؿ��(mm), �ɰ���Ҫ�޸�
		L=N*pix;         %SLM���(mm)
		z0=250         %----�������(mm),
		% z0=input('�������(mm)');
		L0=h*z0/pix;      %�ؽ���ƽ����(mm)
		%%%%%%%%%%%%%%%
		Lf=1/pix;        %Ƶ����
		[u,v]=meshgrid(linspace(-Lf/2,Lf/2,N),linspace(-Lf/2,Lf/2,N));
		H=exp(i*k*z0*sqrt(1-(w*u).^2-(w*v).^2));
		CH=conj(H);
		%%%%%%%%%%%%%%

		Y=double(X);
		U0=Y;
		X0=abs(U0);       %��ʼ�����,�������������
		% figstr=strcat('SLMƽ����=',num2str(L),'mm');
		% figstr0=strcat('��ʼ��ƽ����=',num2str(L0),'mm');
		% figure(1),imshow(X,[]),colormap(gray); xlabel(figstr);title('��ƽ��ͼ��');
		% np=input('��������');
		np=15;
		for p=1:np+1    %��������
				%---------------�����������S-FFT���㿪ʼ
				n=1:N;
				x=-L0/2+L0/N*(n-1);	 					
				y=x;
				[yy,xx] = meshgrid(y,x); 
				Fresnel=exp(-i*k/2/z0*(xx.^2+yy.^2)); %���ű�ʾ������
				f2=U0.*Fresnel;
				Uf=fft2(f2,N,N);
				% Uf=fftshift(Uf);
				x=-L/2+L/N*(n-1);%SLM���ȡ��(mm) 					
				y=x;
				[yy,xx] = meshgrid(y,x); 
				phase=exp(-i*k*z0)/(-i*h*z0)*exp(-i*k/2/z0*(xx.^2+yy.^2));
				Uf=Uf.*phase.*cmask1;
				%�������ý��׷������Ŀǰ��ƽ�������䵽����ƽ����λ��Ĺ���
				Ui=ifft2(fft2(Uf).*CH); %��������ƽ��Ⲩ�����

				%---------------�����������S-FFT�������
				% figstr=strcat('SLM���=',num2str(L),'mm');
				% figure(2),imshow(abs(Uf),[]),colormap(gray); xlabel(figstr);title('����SLMƽ����������ֲ�');

				Phase=angle(Ui)+pi;
				% Phase1=Phase;
				Ih=uint8(Phase/2/pi*255);%�γ�0-255�Ҷȼ�����Ϣͼ
				% figure(3),imshow(Phase,[]),colormap(gray); xlabel(figstr);title('��Ϣͼ');
				%---------------�����������S-IFFT���㿪ʼ
				%������ƽ�濪ʼ��CCDƽ������
				Uii=cos(Phase-pi)+i*sin(Phase-pi); %����ƽ��ֻ������λ����Ϣͼ
				Um=ifft2(fft2(Uii).*H).*mask1; %�м������λ������ĸ���� 
				n=1:N;
				x=-L/2+L/N*(n-1);	 					
				y=x;
				[yy,xx] = meshgrid(y,x); 
				Fresnel=exp(i*k/2/z0*(xx.^2+yy.^2));
				f2=Um.*Fresnel;
				Uf=ifft2(f2,N,N);
				Uf=(Uf);
				x=-L0/2+L0/N*(n-1);%�ؽ���ƽ����ȡ��(mm) 					
				y=x;
				[yy,xx] = meshgrid(y,x); 
				phase=exp(i*k*z0)/(i*h*z0)*exp(i*k/2/z0*(xx.^2+yy.^2));
				Uf=Uf.*phase;

				% figure(4),imshow(abs(Uf),[]),colormap(gray); xlabel(figstr0);title('�������ؽ�����ƽ������ֲ�');
				%---------------������λ���䣬����ԭͼ��������¿�ʼ��һ�ּ���
				Phase=angle(Uf);
				U0=X0.*(cos(Phase)+i*sin(Phase));
		end
		New_Ih=Ih;
		% imwrite(abs(X0),strcat('c:\imgsWan\Train128\',num2str(nn),'.bmp'));
		imwrite(New_Ih,strcat('c:\imgsWanHolo\Train\',num2str(nn),'.jpg'));
end
