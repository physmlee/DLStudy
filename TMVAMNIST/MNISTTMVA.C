/********************************************************
 *                                                     * 
 * MNISTTMVA.C                                         * 
 * Written by Seungmok Lee                             * 
 * Seoul Nat'l Univ.                                   * 
 * Department of Physics and Astronomy                 * 
 * email: physmlee@gmail.com                           * 
 * git : https://github.com/physmlee/DLStudy           * 
 * Date: 2020.01.30                                    * 
 *                                                     * 
 * Tested Enviornment                                  * 
 *   Python		2.7                                    * 
 *   ROOT		6.18/04                                * 
 *   tensorflow	1.14.0                                 * 
 *   keras		2.3.1                                  * 
 * In Ubuntu 18.04 LTS                                 * 
 *                                                     * 
*********************************************************
 *                                                     * 
 *                   INSTRUCTION                       * 
 *                                                     * 
 * This macro is an example multiclass classifier for  * 
 * MNIST dataset using TMVA methods. It loads the data * 
 * file, and then carries out the DL. If the data      * 
 * file is not found, getenates it running             * 
 * 'MNISTtoROOT.py' python macro.                      * 
 *                                                     * 
 * Run this by typing                                  *
 *                                                     * 
 *   >> root MNISTTMVA.C                               * 
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
 * [2] https://root.cern/doc/master/TMVAMulticlass_8C_source.html
 *     TMVA Multiclass Example Code                    * 
 * [3] https://github.com/root-project/root/blob/master/documentation/tmva/UsersGuide/TMVAUsersGuide.pdf
 *     TMVA 4 Users Guide for ROOT >= 6.12 Version     * 
 *                                                     * 
********************************************************/

#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "TMVA/Tools.h"
#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/TMVAMultiClassGui.h"

using namespace TMVA;

void MNISTTMVA()
{
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

	// Setup TMVA
	TMVA::Tools::Instance();
	TString outfileName = "./data/MNISTTMVA.root";
	TFile *output = TFile::Open(outfileName, "RECREATE");
	TMVA::Factory *factory = new TMVA::Factory("TMVAMulticlass", output,
											   "!V:Color:DrawProgressBar:"
											   "!Silent:"				   // You can use silent mode instead. Silent mode has fewer outputs.
											   "Transformations=:"		   // No preprocessing for input variable
											   "AnalysisType=multiclass"); // It is a multiclass classification example

	// Load data trees and variables
	Int_t nb_classes = 10; // number of classes
	Float_t weight = 1.0;  // default weight
	TCut cut = "";		   // no cut
	Int_t pixel = 28 * 28; // number of pixels in one image

	TTree *traintree[10], *testtree[10];
	Char_t trainname[10], testname[10], branchname[10], classname[10];

	for (Int_t i = 0; i < nb_classes; i++)
	{
		sprintf(trainname, "train%d", i);
		traintree[i] = (TTree *)data->Get(trainname); // Load training trees
		sprintf(testname,  "test%d" , i);
		testtree[i]  = (TTree *)data->Get(testname);  // Load testing trees
	}

	TMVA::DataLoader *dataloader = new TMVA::DataLoader("dataset");
	for (Int_t i = 0; i < pixel; i++)
	{
		sprintf(branchname, "image%d", i);
		dataloader->AddVariable(branchname, 'F'); // register all the 784 pixels as variable
	}

	for (Int_t i = 0; i < nb_classes; i++)
	{
		sprintf(classname, "%d", i);
		dataloader->AddTree(traintree[i], classname, weight, cut, TMVA::Types::kTraining); // Add trees specifying their purpose (Training)
		dataloader->AddTree(testtree[i] , classname, weight, cut, TMVA::Types::kTesting ); // Add trees specifying their purpose (Testing)
	}

	dataloader->PrepareTrainingAndTestTree(cut,
										   "!CalcCorrelations:" // Skip calculating decorrelation matrix
										   "NormMode=None:"		// Normalization makes the entry numbers of each class to be equal. It is not our business.
										   "!V");				// No verbose option

	// Model generating start
	TString layoutString("Layout=RELU|512," // First hidden layer with 512 neurons, RELU activation function
						 "RELU|512,"		// Second hidden layer with 512 neurons, RELU activation function
						 "LINEAR|10");		// Final output layer to 10 categories
											// SOFTMAX activaion function is included in the loss function, so we should use LINEAR (identity) activation function in the output layer
	TString training("Repetitions=1,Regularization=None,Multithreading=True,"
					 "Optimizer=ADAM,LearningRate=0.001," // Use Adam optimizer with learning rate 0.001
					 "MaxEpochs=5,BatchSize=128,"		  // Batch size 128 and train 5 times
					 "ConvergenceSteps=100,"			  // Do not use Convergence check
					 "TestRepetitions=1,"				  // Show validation result for every epochs
					 "DropConfig=0.2+0.2+0");			  // Set dropout. 0.2 for the first and second layers, and 0 for the final layer.
	TString nnOptions("!H:!V:VarTransform=:"
					  "ErrorStrategy=MUTUALEXCLUSIVE:" // Use SOFTMAX * CROSSENTROPY loss function
					  "ValidationSize=128");		   // I don't want to split my training dataset to validation dataset, but events more than batchsize must be given to validation dataset.

	TString trainingStrategyString("TrainingStrategy=");
	trainingStrategyString += training; // Use only one training strategy

	nnOptions.Append(":");
	nnOptions.Append(layoutString); // Register the model
	nnOptions.Append(":");
	nnOptions.Append(trainingStrategyString); // Register the model

	// Book method
	factory->BookMethod(dataloader, TMVA::Types::kDL, "MNISTTMVA", nnOptions);

	// Run TMVA
	factory->TrainAllMethods();
	factory->TestAllMethods();
	factory->EvaluateAllMethods();

	output->Close();

	delete factory;
	delete dataloader;
}

// Code Ends Here
