// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include "ort_genai.h"

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
using Duration = std::chrono::duration<double>;

// C++ API Example

void CXX_API(const char* model_path, const std::string& text) {
    auto model = OgaModel::Create(model_path);
    auto tokenizer = OgaTokenizer::Create(*model);
    // auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);

    const std::string prompt = "<|user|>\n" + text + "<|end|>\n<|assistant|>";

    auto sequences = OgaSequences::Create();
    tokenizer->Encode(prompt.c_str(), *sequences);

    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", 2048);
    params->SetInputSequences(*sequences);

    auto generator = OgaGenerator::Create(*model, *params);

    std::string output = "";
    while (!generator->IsDone()) {
        generator->ComputeLogits();
        generator->GenerateNextToken();

        // Show usage of GetOutput
        // std::unique_ptr<OgaTensor> output_logits = generator->GetOutput("logits");

        // Assuming output_logits.Type() is float as it's logits
        // Assuming shape is 1 dimensional with shape[0] being the size
        // auto logits = reinterpret_cast<float*>(output_logits->Data());

        // Print out the logits using the following snippet, if needed
        // auto shape = output_logits->Shape();
        // for (size_t i=0; i < shape[0]; i++)
        //     std::cout << logits[i] << " ";
        // std::cout << std::endl;

        const auto num_tokens = generator->GetSequenceCount(0);
        const auto new_token = generator->GetSequenceData(0)[num_tokens - 1];
        output += tokenizer->Decode(&new_token, num_tokens);
        // std::cout << tokenizer_stream->Decode(new_token) << std::flush;
    }
    std::cout << output << std::endl;
}

// static void print_usage(int /*argc*/, char** argv) {
//   std::cerr << "usage: " << argv[0] << " model_path" << std::endl;
// }

int main(int argc, char* argv[]) {
  // if (argc != 3) {
  //   print_usage(argc, argv);
  //   return -1;
  // }

  // Responsible for cleaning up the library during shutdown
  OgaHandle handle;

  std::cout << "C++ API" << std::endl;
  CXX_API(argv[1], argv[2]);


  return 0;
}