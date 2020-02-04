/********************************************************
 *                                                     * 
 * MNIST_TMVA_CNN.C                                    * 
 * Written by Seungmok Lee                             * 
 * Seoul Nat'l Univ.                                   * 
 * Department of Physics and Astronomy                 * 
 * email: physmlee@gmail.com                           * 
 * git : https://github.com/physmlee/DLStudy           * 
 * Date: 2020.02.04                                    * 
 *                                                     * 
 * Tested Enviornment                                  * 
 *   Python     2.7                                    * 
 *   ROOT       6.18/04                                * 
 *   tensorflow 1.14.0                                 * 
 *   keras      2.3.1                                  * 
 * In Ubuntu 18.04 LTS                                 * 
 *                                                     * 
*********************************************************
 *                                                     * 
 *                   INSTRUCTION                       * 
 *                                                     * 
 * This macro is an example multiclass CNN for MNIST   * 
 * dataset using TMVA kDL methods. It loads the data   * 
 * file, and then carries out CNN. If the data file is * 
 * not found, getenates it running 'MNISTtoROOT.py'    * 
 * python macro.                                       * 
 *                                                     * 
 * Run this by typing                                  *
 *                                                     * 
 *   >> root MNIST_TMVA_CNN.C                          * 
 *                                                     * 
 * Classification result will be saved in 'dataset'    *
 * directory. Please refer to the TMVA Users Guide     * 
 * about how to read the output file.                  * 
 *                                                     * 
*********************************************************
 *                                                     * 
 * Reference                                           * 
 * [1] https://colab.research.google.com/github/AviatorMoser/keras-mnist-tutorial/blob/master/MNIST%20in%20Keras.ipynb#scrollTo=nHLN8vcb6rws
 *     MNIST in Keras Example Code                     * 
 * [2] https://github.com/lmoneta/tmva-tutorial/blob/master/notebooks/TMVA_CNN_Classification.C
 *     TMVA CNN Example Code                    * 
 * [3] https://github.com/root-project/root/blob/master/documentation/tmva/UsersGuide/TMVAUsersGuide.pdf
 *     TMVA 4 Users Guide for ROOT >= 6.12 Version     * 
 *                                                     * 
********************************************************/

void MNIST_TMVA_CNN()
{
    TMVA::Tools::Instance();
    // do enable MT running
    ROOT::EnableImplicitMT(); 

    // Set Output
    auto output = TFile::Open("MNIST_TMVA_CNN_Output.root", "RECREATE");

    // Declare Factory
	TMVA::Factory *factory = new TMVA::Factory("MNIST_TMVA_CNN", output,
											   "V:Color:DrawProgressBar:"
											   "!Silent:"				   // You can use silent mode instead. Silent mode has fewer outputs.
											   "Transformations=:"		   // No preprocessing for input variable
											   "!Correlations:"
                                               "!ROC:"
                                               "AnalysisType=multiclass"); // It is a multiclass classification example

    // Declare DataLoader
	TMVA::DataLoader *dataloader = new TMVA::DataLoader("dataset");
	Int_t pixel = 28 * 28; // number of pixels in one image
    for (Int_t i = 0; i < pixel; i++)
	{
		dataloader->AddVariable(Form("image%d", i), 'F'); // register all the 784 pixels as variable
	}

	// Load MNIST dataset ROOT file
	TString datafilename = "./data/MNIST.root";
	if (gSystem->AccessPathName(datafilename))
	{
		printf("Could not open data file. Trying to download it...\n");
		if (gSystem->AccessPathName("./MNISTtoROOT.py"))
		{
			printf("Cannot find 'MNISTtoROOT.py' macro. Please download it from https://github.com/physmlee/DLStudy\n");
			printf("Exit.\n");
			exit(1);
		}
		Int_t runpy = (Int_t)system("python MNISTtoROOT.py");
		if (runpy == 127 || runpy == -1)
		{
			printf("Could not run 'MNISTtoROOT.py'. Exit.\n");
			exit(1);
		}
	}

	TFile *data = TFile::Open(datafilename, "READ");
	if (!data->IsOpen())
	{
		printf("Data file open error.\n");
		printf("Exit.\n");
		exit(1);
	}

    // Get Trees
	TTree *traintree[10], *testtree[10];
	Int_t nb_classes = 10; // number of classes
	for (Int_t i = 0; i < nb_classes; i++)
	{
		traintree[i] = (TTree *)data->Get(Form("train%d", i)); // Load training trees
		testtree[i]  = (TTree *)data->Get(Form("test%d", i));  // Load testing trees
	}

    // Load Trees
	Float_t weight = 1.0;  // default weight
	TCut cut = "";		   // no cut
	for (Int_t i = 0; i < nb_classes; i++)
	{
		dataloader->AddTree(traintree[i], Form("%d", i), weight, cut, TMVA::Types::kTraining); // Add trees specifying their purpose (Training)
		dataloader->AddTree(testtree[i] , Form("%d", i), weight, cut, TMVA::Types::kTesting ); // Add trees specifying their purpose (Testing)
	}

    // Prepare DataLoader
	dataloader->PrepareTrainingAndTestTree(cut,
										   "!CalcCorrelations:" // Skip calculating decorrelation matrix
										   "NormMode=None:"		// Normalization makes the entry numbers of each class to be equal. It is not our business.
										   "!Correlations:"
                                           "V");				// No verbose option

  /***
   Reference: https://github.com/lmoneta/tmva-tutorial/blob/master/notebooks/TMVA_CNN_Classification.C
   ### Book Convolutional Neural Network in TMVA
   For building a CNN one needs to define 

   -  Input Layout :  number of channels (in this case = 1)  | image height | image width
   -  Batch Layout :  batch size | number of channels | image size = (height*width)

   Then one add Convolutional layers and MaxPool layers. 

   -  For Convolutional layer the option string has to be: 
      - CONV | number of units | filter height | filter width | stride height | stride width | padding height | paddig width | activation function 

      - note in this case we are using a filer 3x3 and padding=1 and stride=1 so we get the output dimension of the conv layer equal to the input

    - For the MaxPool layer: 
       - MAXPOOL  | pool height | pool width | stride height | stride width

   The RESHAPE layer is needed to flatten the output before the Dense layer
   Note that to run the CNN is required to have CPU  or GPU support 
  ***/

    // Generating Model
    TString inputLayoutString("InputLayout=1|28|28");
    TString batchLayoutString("BatchLayout=128|1|784");

    TString convLayer01("CONV|32|3|3|1|1|0|0|RELU");
    TString batchNormLayer01("BNORM|0.99|0.0001"); // Batch normalization layer is not working.

    TString convLayer02("CONV|32|3|3|1|1|0|0|RELU");
    TString batchNormLayer02("BNORM|0.99|0.001"); // Batch normalization layer is not working.
    TString maxPooling02("MAXPOOL|2|2|2|2");

    TString convLayer03("CONV|64|3|3|1|1|0|0|RELU");
    TString batchNormLayer03("BNORM|0.99|0.001"); // Batch normalization layer is not working.

    TString convLayer04("CONV|64|3|3|1|1|0|0|RELU");
    TString batchNormLayer04("BNORM|0.99|0.001"); // Batch normalization layer is not working.
    TString maxPooling04("MAXPOOL|2|2|2|2");
    TString flatten04("RESHAPE|FLAT");

    TString fullyConnLayer05("DENSE|512|RELU");
    TString batchNormLayer05("BNORM|0.99|0.001"); // Batch normalization layer is not working.

    TString fullyConnLayer06("DENSE|10|LINEAR"); // SOFTMAX is included in the loss function.

    TString layoutString("Layout=");
    /*
    Batch normalization layer is not working.
    layoutString += "," + convLayer01 + "," + batchNormLayer01;
    layoutString += "," + convLayer02 + "," + batchNormLayer02 + "," + maxPooling02;
    layoutString += "," + convLayer03 + "," + batchNormLayer03;
    layoutString += "," + convLayer04 + "," + batchNormLayer04 + "," + maxPooling04 + "," + flatten04;
    layoutString += "," + fullyConnLayer05 + "," + batchNormLayer05;
    layoutString += "," + fullyConnLayer06;
    */

    layoutString += "," + convLayer01;
    layoutString += "," + convLayer02 + "," + maxPooling02;
    layoutString += "," + convLayer03;
    layoutString += "," + convLayer04 + "," + maxPooling04 + "," + flatten04;
    layoutString += "," + fullyConnLayer05;
    layoutString += "," + fullyConnLayer06;

    TString dropoutConfig("DropConfig=0.0+0.0+0.0+0.0+0.0+0.2");
    TString optimizer("Optimizer=ADAM,LearningRate=1.e-3");
    TString batchEpochs("BatchSize=128,MaxEpochs=5");
    TString training("Repetitions=1,Regularization=None,Multithreading=True,ConvergenceSteps=100,TestRepetitions=1");
    training += "," + dropoutConfig;
    training += "," + optimizer;
    training += "," + batchEpochs;
    TString trainingStrategyString("TrainingStrategy=");
    trainingStrategyString += training;

    TString cnnOptions("!H:V:ErrorStrategy=MUTUALEXCLUSIVE:VarTransform=None:"
                              "WeightInitialization=XAVIER:"
                              "ValidationSize=128");
    
    cnnOptions.Append(":"); cnnOptions.Append(inputLayoutString);
    cnnOptions.Append(":"); cnnOptions.Append(batchLayoutString);
    cnnOptions.Append(":"); cnnOptions.Append(layoutString);
    cnnOptions.Append(":"); cnnOptions.Append(trainingStrategyString);

    factory->BookMethod(dataloader, TMVA::Types::kDL, "MNIST_TMVA_CNN", cnnOptions);

    factory->TrainAllMethods();
	factory->TestAllMethods();
	factory->EvaluateAllMethods();

	output->Close();
    data->Close();

	delete factory;
	delete dataloader;
}