clear;
close all;
N=128;  
% mask1=exp(i*rand(N,N)*2*pi);
load 'c:\mask1.mat';
cmask1=conj(mask1);
% save ('c:\mask1.mat', 'mask1') ;

chemin='c:\imgsWan\Train\';%图片路径
Image=dir('c:\imgsWan\Train\');
Number=length(Image)-2;
for nn=1:(Number)

		XRGB=imread(strcat(chemin,num2str(nn),'.jpg'));
		X0=rgb2gray(XRGB);                       %彩色图像转换为灰度图像
		% X0=(XRGB);  
		X0=imresize(X0,[128 128],'bicubic');
		% X0=(XRGB);
		[M0,N0]=size(X0);                            %获取灰度图像的像素数大小
		N1=min(M0,N0);
		%相息图取样数, 可按需要修改
		m0=1;                                          %图像在重建周期中的显示比例, 
		X1=imresize(X0,N/N1*m0);
		[M1,N1]=size(X1);
		X=zeros(N,N);
		X(N/2-M1/2+1:N/2+M1/2,N/2-N1/2+1:N/2+N1/2)=X1(1:M1,1:N1);

		h=0.532e-3;      %波长(mm), 可按需要修改
		w=h;
		k=2*pi/h;
		pix=0.0064;      %SLM像素宽度(mm), 可按需要修改
		L=N*pix;         %SLM宽度(mm)
		z0=250         %----衍射距离(mm),
		% z0=input('衍射距离(mm)');
		L0=h*z0/pix;      %重建像平面宽度(mm)
		%%%%%%%%%%%%%%%
		Lf=1/pix;        %频域宽度
		[u,v]=meshgrid(linspace(-Lf/2,Lf/2,N),linspace(-Lf/2,Lf/2,N));
		H=exp(i*k*z0*sqrt(1-(w*u).^2-(w*v).^2));
		CH=conj(H);
		%%%%%%%%%%%%%%

		Y=double(X);
		U0=Y;
		X0=abs(U0);       %初始场振幅,后面叠代运算用
		% figstr=strcat('SLM平面宽度=',num2str(L),'mm');
		% figstr0=strcat('初始物平面宽度=',num2str(L0),'mm');
		% figure(1),imshow(X,[]),colormap(gray); xlabel(figstr);title('物平面图像');
		% np=input('叠代次数');
		np=15;
		for p=1:np+1    %叠代次数
				%---------------菲涅耳衍射的S-FFT计算开始
				n=1:N;
				x=-L0/2+L0/N*(n-1);	 					
				y=x;
				[yy,xx] = meshgrid(y,x); 
				Fresnel=exp(-i*k/2/z0*(xx.^2+yy.^2)); %负号表示逆衍射
				f2=U0.*Fresnel;
				Uf=fft2(f2,N,N);
				% Uf=fftshift(Uf);
				x=-L/2+L/N*(n-1);%SLM宽度取样(mm) 					
				y=x;
				[yy,xx] = meshgrid(y,x); 
				phase=exp(-i*k*z0)/(-i*h*z0)*exp(-i*k/2/z0*(xx.^2+yy.^2));
				Uf=Uf.*phase.*cmask1;
				%接下来用角谱法计算从目前的平面逆衍射到输入平面相位板的过程
				Ui=ifft2(fft2(Uf).*CH); %计算输入平面光波复振幅

				%---------------菲涅耳衍射的S-FFT计算结束
				% figstr=strcat('SLM宽度=',num2str(L),'mm');
				% figure(2),imshow(abs(Uf),[]),colormap(gray); xlabel(figstr);title('到达SLM平面的物光振幅分布');

				Phase=angle(Ui)+pi;
				% Phase1=Phase;
				Ih=uint8(Phase/2/pi*255);%形成0-255灰度级的相息图
				% figure(3),imshow(Phase,[]),colormap(gray); xlabel(figstr);title('相息图');
				%---------------菲涅耳衍射的S-IFFT计算开始
				%从输入平面开始向CCD平面衍射
				Uii=cos(Phase-pi)+i*sin(Phase-pi); %输入平面只保留相位的相息图
				Um=ifft2(fft2(Uii).*H).*mask1; %中间随机相位板后表面的复振幅 
				n=1:N;
				x=-L/2+L/N*(n-1);	 					
				y=x;
				[yy,xx] = meshgrid(y,x); 
				Fresnel=exp(i*k/2/z0*(xx.^2+yy.^2));
				f2=Um.*Fresnel;
				Uf=ifft2(f2,N,N);
				Uf=(Uf);
				x=-L0/2+L0/N*(n-1);%重建像平面宽度取样(mm) 					
				y=x;
				[yy,xx] = meshgrid(y,x); 
				phase=exp(i*k*z0)/(i*h*z0)*exp(i*k/2/z0*(xx.^2+yy.^2));
				Uf=Uf.*phase;

				% figure(4),imshow(abs(Uf),[]),colormap(gray); xlabel(figstr0);title('逆运算重建的物平面振幅分布');
				%---------------保持相位不变，引用原图振幅，重新开始新一轮计算
				Phase=angle(Uf);
				U0=X0.*(cos(Phase)+i*sin(Phase));
		end
		New_Ih=Ih;
		% imwrite(abs(X0),strcat('c:\imgsWan\Train128\',num2str(nn),'.bmp'));
		imwrite(New_Ih,strcat('c:\imgsWanHolo\Train\',num2str(nn),'.jpg'));
end
