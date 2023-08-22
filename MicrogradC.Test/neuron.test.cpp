#include "pch.h"

#include <memory>

#include "..\MikrocradC\neuron.h"
#include "..\MikrocradC\\trace.h"



TEST(TestNeuron, Neuron)
{
	neuron n(2);

	std::vector<std::shared_ptr<value>> x;
	x.push_back(std::make_shared<value>(value(2.0f)));
	x.push_back(std::make_shared<value>(value(3.0f)));

	auto result = n(x); result->set_label(std::string("result"));

	trace(*result);

	std::cout << *result << "\n";
}

TEST(TestNeuron, NeuronInitializerList1)
{
	neuron n({ 0.5f, 0.7f }, -0.3f);

	std::vector<std::shared_ptr<value>> x;
	x.push_back(std::make_shared<value>(value(2.0f, "x_input1")));
	x.push_back(std::make_shared<value>(value(3.0f, "x_input2")));

	auto result = n(x); result->set_label("result");

	trace(*result);

	std::cout << *result << "\n";

	//EXPECT_NEAR(result, 0.997f, 0.01);
}

TEST(TestNeuron, NeuronInitializerList2)
{
	neuron n({ value(0.38f, "init1"), value(-0.7f, "init2") }, value(-0.8f, "bias"));

	std::vector<std::shared_ptr<value>> x;
	x.push_back(std::make_shared<value>(value(2.0f, "x_input1")));
	x.push_back(std::make_shared<value>(value(3.0f, "x_input2")));

	auto result = n(x); result->set_label("result");

	trace(*result);

	std::cout << *result << "\n";

	EXPECT_NEAR(*result, -0.9758, 0.01);
}

TEST(TestNeuron, Layer)
{
	layer layer(2, 3);

	std::vector<std::shared_ptr<value>> x;
	x.push_back(std::make_shared<value>(value(2.0f, "x_input1")));
	x.push_back(std::make_shared<value>(value(3.0f, "x_input2")));

	auto out = layer(x);

	int count = 0;

	for (auto i : out)
	{
		trace(*i);
		std::cout << "\n";
		count++;
	}
}

TEST(TestNeuron, MLP)
{
	std::vector<int> layersizes;

	layersizes.push_back(4);
	layersizes.push_back(4);
	layersizes.push_back(1);

	mlp m(3, layersizes);

	std::vector<std::shared_ptr<value>> x;
	x.push_back(std::make_shared<value>(value(2.0f, std::string("x_input1"))));
	x.push_back(std::make_shared<value>(value(3.0f, std::string("x_input2"))));
	x.push_back(std::make_shared<value>(value(-1.0f, "x_input3")));

	std::vector<std::shared_ptr<value>> result = m(x);

	for (auto val : result)
	{
		std::cout << *val << '\n';
		trace(*val);
		val->backward();
		trace(*val);
	}
}

void make_input(std::vector<std::shared_ptr<value>>& x, int num, const float * ar)
{
	x.clear();

	for (int ii = 0; ii < num; ii++)
	{
		x.push_back(std::make_shared<value>(value(ar[ii], std::string("input") + std::to_string(ii))));
	}
}

TEST(TestNeuron, MLP_2)
{
	std::vector<int> layersizes;

	layersizes.push_back(4);
	layersizes.push_back(4);
	layersizes.push_back(1);

	mlp m(3, layersizes);

	std::vector<std::shared_ptr<value>> input_values[4];
	std::vector<std::shared_ptr<value>> results[4];

	float inputs[4][3] = { {2.0f, 3.0f, -1.0f},
							{3.0f, -1.0f, 0.5f},
							{0.5f, 1.0f, 1.0f},
							{1.0f, 1.0f, -1.0f} };

	
	make_input(input_values[0], 3, inputs[0]);

	std::vector<std::shared_ptr<value>> result = m(input_values[0]);

	for (auto val : result)
	{
		std::cout << *val << '\n';
		trace(*val);
		val->backward();
		trace(*val);
	}
}


TEST(TestNeuron, MLP_MultipleSamples)
{
	std::vector<int> layersizes;

	layersizes.push_back(4);
	layersizes.push_back(4);
	layersizes.push_back(1);

	mlp m(3, layersizes);

	std::vector<std::shared_ptr<value>> input_values[4];
	std::vector<std::shared_ptr<value>> results[4];

	float inputs[4][3] = { {2.0f, 3.0f, -1.0f},
							{3.0f, -1.0f, 0.5f},
							{0.5f, 1.0f, 1.0f},
							{1.0f, 1.0f, -1.0f} };

	float desired_targets[4] = { 1.0f, -1.0f, -1.0f, 1.0f };
	float resultOut[4] = { 0.0f };
	
	for (int ii = 0; ii < 4; ii++)
	{
		make_input(input_values[ii], 3, inputs[ii]);
		results[ii] = m(input_values[ii]);

		for (auto val : results[ii])
		{
			std::cout << *val << '\n';
			resultOut[ii] = *val;
			//trace(*val);
			val->backward();
			//trace(*val);
		}

		results[ii] = m(input_values[ii]);

		for (auto val : results[ii])
		{
			std::cout << *val << '\n';
			resultOut[ii] = *val;
			//trace(*val);
			//val->backward();
			//trace(*val);
		}
	}

	float loss[4];
	float totalLoss = 0.0f;

	for (int ii = 0; ii < 4; ii++)
	{
		float diff = resultOut[ii] - desired_targets[ii];
		loss[ii] = diff * diff;
		std::cout << "loss" << ii << " " << loss[ii] << "\n";
		totalLoss += loss[ii];
	}

	std::cout << "TotalLoss: " << totalLoss << std::endl;
}

TEST(TestNeuron, MLP_Backprop)
{
	std::vector<int> layersizes;

	layersizes.push_back(4);
	layersizes.push_back(4);
	layersizes.push_back(1);

	mlp m(3, layersizes);

	std::vector<std::shared_ptr<value>> input_values[4];
	std::vector<std::shared_ptr<value>> results[4];
	std::vector<std::shared_ptr<value>> loss;

	float inputs[4][3] = { {2.0f, 3.0f, -1.0f},
							{3.0f, -1.0f, 0.5f},
							{0.5f, 1.0f, 1.0f},
							{1.0f, 1.0f, -1.0f} };

	value desired_targets[4] = { value(1.0f, "target0"),
								value(-1.0f, "target1"),
								value(-1.0f, "target2"),
								value(1.0f , "target3") };

	for (int ii = 0; ii < 4; ii++)
	{
		make_input(input_values[ii], 3, inputs[ii]);
		results[ii] = m(input_values[ii]);
		trace(*results[ii][0]);
	}

	
	float resultOut[4] = { 0.0f };

	int count = 0;

	float floss = 0.0;

	for (auto result : results)
	{
		for (auto r : result)
		{
			std::cout << "TARGET: " << *r << '\n';
			resultOut[count] = *r;
		

			floss += ((resultOut[count] - desired_targets[count]) * (resultOut[count] - desired_targets[count]));
			count++;
		}
	}

	std::cout << "check calculated loss: " << floss << std::endl;

}


TEST(TestNeuron, MLP_Backprop_loss)
{
	std::vector<int> layersizes;

	layersizes.push_back(4);
	layersizes.push_back(4);
	layersizes.push_back(1);

	mlp m(3, layersizes);

	std::vector<std::shared_ptr<value>> input_values[4];
	std::vector<std::shared_ptr<value>> results[4];

	float inputs[4][3] = { {2.0f, 3.0f, -1.0f},
							{3.0f, -1.0f, 0.5f},
							{0.5f, 1.0f, 1.0f},
							{1.0f, 1.0f, -1.0f} };

	value desired_targets[4] = { value(1.0f, "target0"),
								value(-1.0f, "target1"),
								value(-1.0f, "target2"),
								value(1.0f , "target3") };

	make_input(input_values[0], 3, inputs[0]);
	results[0] = m(input_values[0]);
	auto localLoss0 = std::make_shared<value>(*results[0][0] - desired_targets[0]); localLoss0->set_label("localLoss0");
	auto localSum0 = std::make_shared<value>(*localLoss0 * (*localLoss0)); localSum0->set_label("localSum0");

	make_input(input_values[1], 3, inputs[1]);
	results[1] = m(input_values[1]);
	auto localLoss1 = std::make_shared<value>(*results[1][0] - desired_targets[1]); localLoss0->set_label("localLoss1");
	auto localSum1 = std::make_shared<value>(*localLoss1 * (*localLoss1)); localSum1->set_label("localSum1");

	auto localSum01 = std::make_shared<value>(*localSum0 + (*localSum1)); localSum01->set_label("localSum01");
	

	make_input(input_values[2], 3, inputs[2]);
	results[2] = m(input_values[2]);
	auto localLoss2 = std::make_shared<value>(*results[2][0] - desired_targets[2]); localLoss0->set_label("localLoss2");
	auto localSum2 = std::make_shared<value>(*localLoss2 * (*localLoss2)); localSum2->set_label("localSum2");


	make_input(input_values[3], 3, inputs[3]);
	results[3] = m(input_values[3]);
	auto localLoss3 = std::make_shared<value>(*results[3][0] - desired_targets[3]); localLoss3->set_label("localLoss3");
	auto localSum3 = std::make_shared<value>(*localLoss3 * (*localLoss3)); localSum3->set_label("localSum3");

	auto localSum23 = std::make_shared<value>(*localSum2 + *localSum3); localSum23->set_label("localSum23");
	auto localSum0123 = std::make_shared<value>(*localSum01 + (*localSum23)); localSum0123->set_label("localSum0123");

	localSum0123->set_grad(1.0f);
	localSum0123->backward();
	trace(*localSum0123);


	float resultOut[4] = { 0.0f };

	int count = 0;

	float floss = 0.0;

	for (auto result : results)
	{
		for (auto r : result)
		{
			std::cout << *r << '\n';
			resultOut[count] = *r;


			floss += ((resultOut[count] - desired_targets[count]) * (resultOut[count] - desired_targets[count]));
			count++;
		}
	}

	std::cout << "calculated loss: " << *localSum0123 << std::endl;
	std::cout << "check calculated loss: " << floss << std::endl;
}

TEST(TestNeuron, MLP_Backprop_loss_iterative)
{
	std::vector<int> layersizes;

	layersizes.push_back(4);
	layersizes.push_back(4);
	layersizes.push_back(1);

	mlp m(3, layersizes);

	std::vector<std::shared_ptr<value>> input_values[4];
	std::vector<std::shared_ptr<value>> results[4];
	std::vector<std::shared_ptr<value>> localLoss;
	std::vector<std::shared_ptr<value>> localSum;

	float inputs[4][3] = { {2.0f, 3.0f, -1.0f},
							{3.0f, -1.0f, 0.5f},
							{0.5f, 1.0f, 1.0f},
							{1.0f, 1.0f, -1.0f} };

	value desired_targets[4] = { value(1.0f, "target0"),
								value(-1.0f, "target1"),
								value(-1.0f, "target2"),
								value(1.0f , "target3") };


	for (int ii = 0; ii < 4; ii++)
	{
		make_input(input_values[ii], 3, inputs[ii]);
		results[ii] = m(input_values[ii]);
		localLoss.push_back(std::make_shared<value>(*results[ii][0] - desired_targets[ii])); localLoss.back()->set_label(std::string("localLoss") + std::to_string(ii));
		localSum.push_back(std::make_shared<value>(*(localLoss[ii]) * (*localLoss[ii])));  localSum.back()->set_label(std::string("localSum") + std::to_string(ii));
	}

	auto localSum01 = std::make_shared<value>(*localSum[0] + *localSum[1]); localSum01->set_label("localSum01");
	auto localSum23 = std::make_shared<value>(*localSum[2] + *localSum[3]); localSum23->set_label("localSum23");
	auto localSum0123 = std::make_shared<value>(*localSum01 + *localSum23); localSum0123->set_label("localSum0123");


	localSum0123->backward();
	trace(*localSum0123);


	float resultOut[4] = { 0.0f };

	int count = 0;

	float floss = 0.0;

	for (auto result : results)
	{
		for (auto r : result)
		{
			std::cout << *r << '\n';
			resultOut[count] = *r;


			floss += ((resultOut[count] - desired_targets[count]) * (resultOut[count] - desired_targets[count]));
			count++;
		}
	}

	std::cout << "calculated loss: " << *localSum0123 << std::endl;
	std::cout << "check calculated loss: " << floss << std::endl;
}

TEST(TestNeuron, MLP_Backprop_parameters)
{
	std::vector<int> layersizes;

	layersizes.push_back(4);
	layersizes.push_back(4);
	layersizes.push_back(1);

	mlp m(3, layersizes);

	std::vector<std::shared_ptr<value>> input_values[4];
	std::vector<std::shared_ptr<value>> results[4];
	std::vector<std::shared_ptr<value>> localLoss;
	std::vector<std::shared_ptr<value>> localSum;

	float inputs[4][3] = { {2.0f, 3.0f, -1.0f},
							{3.0f, -1.0f, 0.5f},
							{0.5f, 1.0f, 1.0f},
							{1.0f, 1.0f, -1.0f} };

	value desired_targets[4] = { value(1.0f, "target0"),
								value(-1.0f, "target1"),
								value(-1.0f, "target2"),
								value(1.0f , "target3") };


	for (int ii = 0; ii < 4; ii++)
	{
		make_input(input_values[ii], 3, inputs[ii]);
		results[ii] = m(input_values[ii]);
		localLoss.push_back(std::make_shared<value>(*results[ii][0] - desired_targets[ii])); localLoss.back()->set_label(std::string("localLoss") + std::to_string(ii));
		localSum.push_back(std::make_shared<value>(*(localLoss[ii]) * (*localLoss[ii])));  localSum.back()->set_label(std::string("localSum") + std::to_string(ii));
	}

	auto localSum01 = std::make_shared<value>(*localSum[0] + *localSum[1]); localSum01->set_label("localSum01");
	auto localSum23 = std::make_shared<value>(*localSum[2] + *localSum[3]); localSum23->set_label("localSum23");
	auto localSum0123 = std::make_shared<value>(*localSum01 + *localSum23); localSum0123->set_label("localSum0123");


	localSum0123->set_grad(1.0);
	localSum0123->backward();

	

	trace(*localSum0123);


	float resultOut[4] = { 0.0f };

	int count = 0;

	float floss = 0.0;

	for (auto result : results)
	{
		for (auto r : result)
		{
			std::cout << *r << '\n';
			resultOut[count] = *r;


			floss += ((resultOut[count] - desired_targets[count]) * (resultOut[count] - desired_targets[count]));
			count++;
		}
	}

	std::cout << "calculated loss: " << *localSum0123 << std::endl;
	std::cout << "check calculated loss: " << floss << std::endl;

	auto params = m.parameters();

	std::cout << "size parameters: " << params.size() << std::endl;

	for (auto ii : params)
	{
		std::cout << ii->label() << ", data: " << *ii << ", grad: " << ii->grad() << '\n';
	}

}

TEST(TestNeuron, MLP_Backprop_parameters_training)
{
	std::vector<int> layersizes;

	layersizes.push_back(4);
	layersizes.push_back(4);
	layersizes.push_back(1);

	mlp m(3, layersizes);

	std::vector<std::shared_ptr<value>> input_values[4];
	std::vector<std::shared_ptr<value>> results[4];
	std::vector<std::shared_ptr<value>> localLoss;
	std::vector<std::shared_ptr<value>> localSum;

	float inputs[4][3] = { {2.0f, 3.0f, -1.0f},
							{3.0f, -1.0f, 0.5f},
							{0.5f, 1.0f, 1.0f},
							{1.0f, 1.0f, -1.0f} };

	value desired_targets[4] = { value(1.0f, "target0"),
								value(-1.0f, "target1"),
								value(-1.0f, "target2"),
								value(1.0f , "target3") };

	for (int jj = 0; jj < 500; jj++)
	{
		localLoss.clear();
		localSum.clear();

		for (int ii = 0; ii < 4; ii++)
		{
			make_input(input_values[ii], 3, inputs[ii]);
			results[ii] = m(input_values[ii]);
			localLoss.push_back(std::make_shared<value>(*results[ii][0] - desired_targets[ii])); localLoss.back()->set_label(std::string("localLoss") + std::to_string(ii));
			localSum.push_back(std::make_shared<value>(*(localLoss[ii]) * (*localLoss[ii])));  localSum.back()->set_label(std::string("localSum") + std::to_string(ii));
		}

		auto localSum01 = std::make_shared<value>(*localSum[0] + *localSum[1]); localSum01->set_label("localSum01");
		auto localSum23 = std::make_shared<value>(*localSum[2] + *localSum[3]); localSum23->set_label("localSum23");
		auto localSum0123 = std::make_shared<value>(*localSum01 + *localSum23); localSum0123->set_label("localSum0123");

		//trace(*localSum0123);

		for (auto result : results)
		{
			for (auto r : result)
			{
				std::cout << *r << '\n';
			}
		}

		std::cout << "calculated loss: " << *localSum0123 << std::endl;

		localSum0123->set_grad(1.0);
		localSum0123->backward();

		auto params = m.parameters();

		std::cout << "size parameters: " << params.size() << std::endl;

		for (auto ii : params)
		{
			//std::cout << ii->label() << ", data: " << *ii << ", grad: " << ii->grad() << '\n';
			ii->setData(*ii - 0.05f * ii->grad());
			ii->set_grad(0.0f);
		}
	}



	//trace(*localSum0123);

	

	

}




