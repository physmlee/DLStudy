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
void MNISTTMVA(  )
{
   // Load MNIST dataset ROOT file
   TString datafilename = "./data/MNIST.root";
   TFile* data = TFile::Open( datafilename, "READ" );

   // Setup TMVA
   // This loads the library
   TMVA::Tools::Instance();
   TString outfileName = "./data/MNISTTMVA.root";
   TFile* output = TFile::Open( outfileName, "RECREATE" );
   TMVA::Factory *factory = new TMVA::Factory( "TMVAMulticlass", output,
                                               "!V:Color:DrawProgressBar:"
                                               "!Silent:"
                                               "Transformations=:"
                                               "AnalysisType=multiclass" );

   // Load data trees and variables
   Int_t nb_classes = 10;
   Int_t pixel = 28*28;
   Float_t weight = 1.0;
   TCut cut = "";
   TTree* traintree[10], testtree[10];
   Char_t trainname[10], testname[10], branchname[10];

   for( Int_t i = 0; i < nb_classes; i++ )
   {
      sprintf( trainname, "train%d", i );
      traintree[i] = ( TTree* ) data->Get( trainname );
      sprintf( testname , "test%d" , i );
      testtree[i]  = ( TTree* ) data->Get( testname  );
   }

   TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset");
   for( Int_t i = 0; i < pixel; i++ )
   {
      sprintf( branchname, "image%d", i );
      dataloader->AddVariable( branchname, 'F' );
   }

}
