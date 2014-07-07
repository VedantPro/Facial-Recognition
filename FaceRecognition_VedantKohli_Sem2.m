function varargout = FaceRecognition_VedantKohli_Sem2(varargin)

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @FaceRecognition_VedantKohli_Sem2_OpeningFcn, ...
                   'gui_OutputFcn',  @FaceRecognition_VedantKohli_Sem2_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before FaceRecognition_VedantKohli_Sem2 is made visible.
function FaceRecognition_VedantKohli_Sem2_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to FaceRecognition_VedantKohli_Sem2 (see VARARGIN)

% Choose default command line output for FaceRecognition_VedantKohli_Sem2
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes FaceRecognition_VedantKohli_Sem2 wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = FaceRecognition_VedantKohli_Sem2_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in Browse.
function Browse_Callback(hObject, eventdata, handles)

%getting input file to facial recognition
[filename pathname] = uigetfile('*.bmp','Select file for Recognition');             
set(handles.FileName,'String',filename);

%displaying input image
set(handles.SImage,'Visible','On');
set(handles.Check,'Visible','On');
input = strcat(pathname,filename);
imshow(input,'Parent',handles.axes1);           
handles.input = input;

guidata(hObject,handles);



% --- Executes on button press in Check.
function Check_Callback(hObject, eventdata, handles)

%converting input image into N1*N2X1 column matrix ( face vector ) 
Input = imread(handles.input);
InImage=reshape(double(Input)',handles.irow*handles.icol,1);  
temp=InImage;

%Normalising the face vector
temp=(temp-handles.m);
NormImage = temp;
Difference = temp;

%getting the Weights for the input normalised face vector
p = [];
aa=size(handles.u,2);
for i = 1:aa
    pare = dot(NormImage,handles.u(:,i));
    p = [p; pare];
end

%recontructing the image for weights
ReshapedImage = handles.m + handles.u(:,1:aa)*p;    %m is the mean image, u is the eigenvector
ReshapedImage = reshape(ReshapedImage,handles.icol,handles.irow);
ReshapedImage = ReshapedImage';

%displaying the recontructed image
set(handles.Reconst,'Visible','On');
imshow(ReshapedImage,[],'Parent',handles.axes4);

%finding the weighted input face
InImWeight = [];
for i=1:size(handles.u,2)
    t = handles.u(:,i)';
    WeightOfInputImage = dot(t,Difference');
    InImWeight = [InImWeight; WeightOfInputImage];
end

%displaying the weight input face
set(handles.Weight,'Visible','On');
ll = 1:size(handles.S,2);
axes(handles.axes5);
stem(ll,InImWeight,'Parent',handles.axes5);

%finding  Euclidean distance
set(handles.Euc,'Visible','On');
e=[];
for i=1:size(handles.omega,2)
    q = handles.omega(:,i);
    DiffWeight = InImWeight-q;
    mag = norm(DiffWeight);
    e = [e mag];
end

%displaying the Euclidean distance
kk = 1:size(e,2);
set(handles.min,'Visible','On');
set(handles.max,'Visible','On');
stem(kk,e,'Parent',handles.axes6);
set(handles.Val1,'String',num2str(max(e)));
set(handles.Val2,'String',num2str(min(e)));



% --- Executes on button press in Directory.
function Directory_Callback(hObject, eventdata, handles)

%getting the directory for face database
directoryname = uigetdir('','Select Directory');
set(handles.Direc,'String',directoryname);      %directory string
handles.directoryname = directoryname;

Bar = waitbar(0,'Computing');

set(handles.MonData,'Visible','On');
set(handles.NFace,'Visible','On');
set(handles.EigFaces,'Visible','On');
set(handles.NormImag,'Visible','Off');
set(handles.DatImag,'Visible','On');

data = dir(strcat(directoryname,'\*.bmp'));
handles.data = data;

%establishing a matrix for images 
S=[];

%converting the database images into N1*N2X1 column matrix
 for i=1:length(data)
     I=strcat(directoryname,'\g',int2str(i),'.bmp');
     eval('I=imread(I);');
    [irow icol]=size(I);             % get the number of rows (N1) and columns (N2)
    temp=reshape(I',irow*icol,1);     %creates a (N1*N2)x1 matrix
    S=[S temp];                      %X is a N1*N2xM matrix after finishing the sequence
    waitbar(i/140,Bar);    
 end
 
% reading images and constructing montage(array of images to display)
K=cell(1,20);
for k=1:20
    K{k}=strcat('g',int2str(k),'.bmp');
    waitbar((20+i)/140,Bar);
end

handles.I = I;
handles.irow = irow;
handles.icol = icol;

%showing the montage of images
montage(K,'Parent',handles.axes2);                  

% calculating the mean face
m= mean(S,2);

%finding the normalise face images
%This is done to reduce the error due to lighting conditions.
normalimg=[];
tm = double(m);
for i=1:size(S,2)
    temp = double(S(:,i)) - tm;
    img=reshape(temp,icol,irow);
    img=img';
    img=mat2gray(img);
    normalimg(:,:,1,i)=img;
    waitbar((40+i)/140,Bar);
end


handles.normalimg = normalimg;

%displaying the noremalized faces
img=reshape(m,icol,irow);    %takes the N1*N2x1 vector and creates a N2xN1 matrix
img=img';                   %creates a N1xN2 matrix by transposing the image.
set(handles.AFace,'Visible','On');
handles.m = m;
MFace = img;
imshow(MFace,[],'Parent',handles.axes3);

% Change image for manipulation
dbx=[];   % A matrix
for i=1:size(S,2)
    temp=double(S(:,i));
    dbx=[dbx temp];
end

%Covariance matrix C=AA', L=A'A
A=dbx;
L=A'*A;
% vv are the eigenvector for L
% dd are the eigenvalue for both L=A'*A and C=A*A';
[vv dd]=eig(L);

% Sort and eliminate those whose eigenvalue is zero
v=[];
d=[];
for i=1:size(vv,2)
    if(dd(i,i)>1e-4)
        v=[v vv(:,i)];
        d=[d dd(i,i)];
    end
    waitbar((60+i)/140,Bar);
end
 
%sort,  will return an ascending sequence
 [B index]=sort(d);
 ind=zeros(size(index));
 dtemp=zeros(size(index));
 vtemp=zeros(size(v));
 len=length(index);
 for i=1:len
    dtemp(i)=B(len+1-i);
    ind(i)=len+1-index(i);
    vtemp(:,ind(i))=v(:,i);
 end
 d=dtemp;
 v=vtemp;
 
 %Normalization of eigenvectors
 for i=1:size(v,2)       
   kk=v(:,i);
   temp=norm(kk);
   v(:,i)=v(:,i)./temp;
   waitbar((80+i)/140,Bar);
 end

%Eigenvectors of C matrix
u=[];
for i=1:size(v,2)
    temp=sqrt(d(i));
    u=[u (dbx*v(:,i))./temp];
    waitbar((100+i)/140,Bar);
end

%Normalization of eigenvectors
for i=1:size(u,2)
   kk=u(:,i);
   temp=norm(kk);
	u(:,i)=u(:,i)./temp;
    waitbar((120+i)/140,Bar);
end

close(Bar);

% show eigenfaces;
faceimg=[];
for i=1:size(u,2)
    img=reshape(u(:,i),icol,irow);
    img=img';
    img=mat2gray(img);
    faceimg(:,:,1,i)=img;
end
handles.faceimg = faceimg;

% Find the weight of each face in the training set.
omega = [];
for h=1:size(dbx,2)
    WW=[];    
    for i=1:size(u,2)
        t = u(:,i)';    
        WeightOfImage = dot(t,dbx(:,h)');
        WW = [WW; WeightOfImage];
    end
    omega = [omega WW];
end

handles.S = S;
handles.omega = omega;
handles.u = u;

guidata(hObject,handles);



% --- Executes on button press in MonData.
function MonData_Callback(hObject, eventdata, handles)

%function for displaying the Montage of database images
set(handles.NormImag,'Visible','Off');
set(handles.DatImag,'Visible','On');
set(handles.EFaces,'Visible','Off');

K = cell(1,length(handles.data));
for k=1:size(handles.data)
    K{k}=strcat(handles.directoryname,'\g',int2str(k),'.bmp');
end
montage(K,'Parent',handles.axes2);


% --- Executes on button press in NFace.
function NFace_Callback(hObject, eventdata, handles)

%function for displaying the normalised face images
set(handles.DatImag,'Visible','Off');
set(handles.NormImag,'Visible','On');
set(handles.EFaces,'Visible','Off');

montage(handles.normalimg,[],'Parent',handles.axes2);


% --- Executes on button press in EigFaces.
function EigFaces_Callback(hObject, eventdata, handles)

%function for displaying the Eigen Vectors or the Eigen faces or Principal Components
set(handles.DatImag,'Visible','Off');
set(handles.NormImag,'Visible','Off');
set(handles.EFaces,'Visible','On');

montage(handles.faceimg,'Parent',handles.axes2);
