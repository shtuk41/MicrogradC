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
	}
}

TEST(TestNeuron, MLP_Defined)
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
	}
}


