#include "pch.h"
#include <value.h>
#include <trace.h>

constexpr float epsilon = 0.00001f;

TEST(TestValue, Addition) 
{
	value v(4.3f, "v");

	value d(5.5f, "d");

	value c = d + v;

	c.set_label("c");

	std::cout << "New Value is " << c << std::endl;

	ASSERT_FLOAT_EQ(c, 9.8f);
	//EXPECT_TRUE(true);
}

TEST(TestValue, Multiplication)
{
	value v(4.3f, "v");

	value d(5.5f, "d");

	value c = d * v;

	c.set_label("c");

	std::cout << "New Value is " << c << std::endl;

	ASSERT_FLOAT_EQ(c, 23.65f);
	//EXPECT_TRUE(true);
}

TEST(TestValue, MultAddExpression1)
{
	value a(2.0f, "a");

	value b(-3.0f, "b");

	value c(10.0, "c");

	value result = a * b + c;
	result.set_label("result");

	std::cout << "New Value is " << result << std::endl;

	ASSERT_FLOAT_EQ(result, 4.0f);
	//EXPECT_TRUE(true);
}

TEST(TestValue, NaturalLog)
{
	value a(9.0f, "a");
	value result = a.log();

	result.set_label("result");
	std::cout << "Result is " << result << std::endl;

	ASSERT_NEAR(result, 2.197, 0.01);
}

TEST(TestValue, BasicTrace)
{
	value a(2.0f, "a");
	value b(-3.0f, "b");
	value c(10.0, "c");
	value d = a * b; d.set_label("d");
	value result = d + c;
	result.set_label("result");
	trace(result);
	ASSERT_FLOAT_EQ(result, 4.0f);
}

TEST(TestValue, DerivativeCalculations_da)
{
	std::cout << "\n First \n";
	float h = 0.001f;

	value a(2.0f, "a");
	value b(-3.0f, "b");
	value c(10.0, "c");
	value e = a * b; e.set_label("e");
	value d = e + c; d.set_label("d");
	value f(-2.0f, "f");
	value L = d * f; L.set_label("L");
	float L1 = L;

	trace(L);

	std::cout << "\n Second \n";

	a = value(2.0f + h, "a");
	b = value(-3.0f, "b");
	c = value(10.0f, "c");
	e = a * b; e.set_label("e");
	d = e + c; d.set_label("d");
	f = value(-2.0f, "f");
	L = d * f; L.set_label("L");
	float L2 = L;

	trace(L);

	float dL = (L2 - L1) / h;

	std::cout << "Derivative: " << dL << '\n';
	EXPECT_NEAR(dL, 6.0f, 0.01);

	//ASSERT_FLOAT_EQ(result, 4.0f);
}

TEST(TestValue, DerivativeCalculations_dL)
{
	std::cout << "\n First \n";

	float h = 0.001f;

	value a(2.0f, "a");
	value b(-3.0f, "b");
	value c(10.0, "c");
	value e = a * b; e.set_label("e");
	value d = e + c; d.set_label("d");
	value f(-2.0f, "f");
	value L = d * f; L.set_label("L");
	float L1 = L;

	trace(L);

	std::cout << "\n Second \n";

	a = value(2.0f, "a");
	b = value(-3.0f, "b");
	c = value(10.0f, "c");
	e = a * b; e.set_label("e");
	d = e + c; d.set_label("d");
	f = value(-2.0f, "f");
	L = d * f; L.set_label("L");
	float L2 = (float)L + h;

	trace(L);

	float dL = (L2 - L1) / h;

	std::cout << "Derivative: " << dL << '\n';

	EXPECT_NEAR(dL, 1.0f, 0.01);
}

TEST(TestValue, DerivativeCalculations_df)
{
	std::cout << "\n First \n";

	float h = 0.001f;

	value a(2.0f, "a");
	value b(-3.0f, "b");
	value c(10.0, "c");
	value e = a * b; e.set_label("e");
	value d = e + c; d.set_label("d");
	value f(-2.0f, "f");
	value L = d * f; L.set_label("L");
	float L1 = L;

	trace(L);

	std::cout << "\n Second \n";

	a = value(2.0f, "a");
	b = value(-3.0f, "b");
	c = value(10.0f, "c");
	e = a * b; e.set_label("e");
	d = e + c; d.set_label("d");
	f = value(-2.0f + h, "f");
	L = d * f; L.set_label("L");
	float L2 = (float)L;

	trace(L);

	float dL = (L2 - L1) / h;

	std::cout << "Derivative: " << dL << '\n';

	EXPECT_NEAR(dL, 4.0f, 0.01);
}

TEST(TestValue, DerivativeCalculations_dd)
{
	std::cout << "\n First \n";

	float h = 0.001f;

	value a(2.0f, "a");
	value b(-3.0f, "b");
	value c(10.0, "c");
	value e = a * b; e.set_label("e");
	value d = e + c; d.set_label("d");
	value f(-2.0f, "f");
	value L = d * f; L.set_label("L");
	float L1 = L;

	trace(L);

	std::cout << "\n Second \n";

	a = value(2.0f, "a");
	b = value(-3.0f, "b");
	c = value(10.0f, "c");
	e = a * b; e.set_label("e");
	d = e + c; d.set_label("d");
	d = d + h;
	f = value(-2.0f, "f");
	L = d * f; L.set_label("L");
	float L2 = (float)L;

	trace(L);

	float dL = (L2 - L1) / h;

	std::cout << "Derivative: " << dL << '\n';

	EXPECT_NEAR(dL, -2.0f, 0.01);
}

TEST(TestValue, Neuron)
{
	//inputs x1, x2
	auto x1 = value(2.0f, "x1");
	auto x2 = value(0.0f, "x2");
	//weights w1, w2
	auto w1 = value(-3.0f, "w1");
	auto w2 = value(1.0f, "w2");
	//bias of the neuron
	auto b = value(6.7f, "b");
	//x1*w1 + x2 * w2 + b
	auto x1w1 = x1 * w1; x1w1.set_label("x1*w1");
	auto x2w2 = x2 * w2; x2w2.set_label("x2*w2");
	auto x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.set_label("x1*w1 + x2*w2");
	auto n = x1w1x2w2 + b; n.set_label("n");
	auto o = n.tanh(); o.set_label("o");

	
	o.backward();
	trace(o);


	EXPECT_NEAR(o, 0.6044f, 0.01);
}

TEST(TestValue, Neuron_bias8)
{
	//inputs x1, x2
	auto x1 = value(2.0f, "x1");
	auto x2 = value(0.0f, "x2");
	//weights w1, w2
	auto w1 = value(-3.0f, "w1");
	auto w2 = value(1.0f, "w2");
	//bias of the neuron
	auto b = value(8.0f, "b");
	//x1*w1 + x2 * w2 + b
	auto x1w1 = x1 * w1; x1w1.set_label("x1*w1");
	auto x2w2 = x2 * w2; x2w2.set_label("x2*w2");
	auto x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.set_label("x1*w1 + x2*w2");
	auto n = x1w1x2w2 + b; n.set_label("n");
	auto o = n.tanh(); o.set_label("o");

	trace(o);

	EXPECT_NEAR(o, 0.9640, 0.01);
}

TEST(TestValue, Neuron_bias6_8813735870195432)
{
	//inputs x1, x2
	auto x1 = value(2.0f, "x1");
	auto x2 = value(0.0f, "x2");
	//weights w1, w2
	auto w1 = value(-3.0f, "w1");
	auto w2 = value(1.0f, "w2");
	//bias of the neuron
	auto b = value(6.8813735870195432f, "b");
	//x1*w1 + x2 * w2 + b
	auto x1w1 = x1 * w1; x1w1.set_label("x1*w1");
	auto x2w2 = x2 * w2; x2w2.set_label("x2*w2");
	auto x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.set_label("x1*w1 + x2*w2");
	auto n = x1w1x2w2 + b; n.set_label("n");
	auto o = n.tanh(); o.set_label("o");
	o.set_grad(1.0f);

	trace(o);

	EXPECT_NEAR(o, 0.7071f, 0.01);
}

TEST(TestValue, Neuron_backprop)
{
	//inputs x1, x2
	auto x1 = value(2.0f, "x1");
	auto x2 = value(0.0f, "x2");
	//weights w1, w2
	auto w1 = value(-3.0f, "w1");
	auto w2 = value(1.0f, "w2");
	//bias of the neuron
	auto b = value(6.8813735870195432f, "b");
	//x1*w1 + x2 * w2 + b
	auto x1w1 = x1 * w1; x1w1.set_label("x1*w1");
	auto x2w2 = x2 * w2; x2w2.set_label("x2*w2");
	auto x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.set_label("x1*w1 + x2*w2");
	auto n = x1w1x2w2 + b; n.set_label("n");
	auto o = n.tanh(); o.set_label("o");

	o.backward();
	
	trace(o);

	EXPECT_NEAR(w2.grad(), 0.0f, 0.01f);
	EXPECT_NEAR(x2.grad(), 0.5f, 0.01f);
	EXPECT_NEAR(w1.grad(), 1.0f, 0.01f);
	EXPECT_NEAR(x1.grad(), -1.5f, 0.01f);
}

TEST(TestValue, Neuron_nodeusedmorethanonce_bug)
{
	auto a = value(3.0f, "a");
	auto b = a + a;
	
	b.backward();

	trace(b);

	EXPECT_NEAR(a.grad(), 2.0f, 0.01f);
}

TEST(TestValue, Neuron_nodeusedmorethanonce_bug2)
{
	auto a = value(-2.0f, "a");
	auto b = value(3.0f, "b");
	auto d = a * b;	d.set_label("d");
	auto e = a + b; e.set_label("e");
	auto f = d * e; f.set_label("f");

	f.backward();

	trace(f);

	EXPECT_NEAR(a.grad(), -3.0f, 0.01f);
	EXPECT_NEAR(b.grad(), -8.0f, 0.01f);
}

TEST(TestValue, Division)
{
	auto a = value(2.0f, "a");
	auto b = value(4.0f, "b");
	auto c = a / b; c.set_label("c");

	EXPECT_NEAR(c, 0.5f, 0.01f);
}

TEST(TestValue, Subtraction)
{
	auto a = value(2.0f, "a");
	auto b = value(4.0f, "b");
	auto c = a - b; c.set_label("c");

	EXPECT_NEAR(c, -2.0f, 0.01f);
}

TEST(TestValue, Neuron_backprop_tanh_explicit)
{
	//inputs x1, x2
	auto x1 = value(2.0f, "x1");
	auto x2 = value(0.0f, "x2");
	//weights w1, w2
	auto w1 = value(-3.0f, "w1");
	auto w2 = value(1.0f, "w2");
	//bias of the neuron
	auto b = value(6.8813735870195432f, "b");
	//x1*w1 + x2 * w2 + b
	auto x1w1 = x1 * w1; x1w1.set_label("x1*w1");
	auto x2w2 = x2 * w2; x2w2.set_label("x2*w2");
	auto x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.set_label("x1*w1 + x2*w2");
	auto n = x1w1x2w2 + b; n.set_label("n");

	//auto o = n.tanh(); o.set_label("o");

	value v2 = value(2.0f, "v2");
	value v1 = value(1.0f, "v1");

	value e = n * v2; e.set_label("e");
	value e1 = e.exp(); e1.set_label("e1");
	value e2 = e.exp(); e2.set_label("e2");
	value e3 = e1 - v1; e3.set_label("e3");
	value e4 = e2 + v1; e4.set_label("e4");
	value e5 = value(-1.0f, "e5");
	value e6 = e4.pow(e5); e6.set_label("e6");

	value o = e3 * e6; o.set_label("o");

	o.backward();

	trace(o);

	EXPECT_NEAR(w2.grad(), 0.0f, 0.01f);
	EXPECT_NEAR(x2.grad(), 0.5f, 0.01f);
	EXPECT_NEAR(w1.grad(), 1.0f, 0.01f);
	EXPECT_NEAR(x1.grad(), -1.5f, 0.01f);
}

TEST(TestValue, DivisionBackward)
{

	auto x1 = value(2.0f, "x1");
	auto x2 = value(4.0f, "x2");
	auto x3 = value(-3.0f, "x3");

	auto x4 = x2 / x3; x4.set_label("x4");
	auto x5 = x1 + x4; x5.set_label("x5");


	x5.set_grad(1.0f);
	x5.backward();
	trace(x5);

	std::cout << "result: " << x5 << std::endl;


	EXPECT_NEAR(x5, 0.667f, 0.01);
}

TEST(TestValue, LogBackward)
{

	auto x1 = value(2.0f, "x1");
	auto x2 = value(4.0f, "x2");
	auto x3 = x2.log(); x3.set_label("x3");
	auto x4 = x1 + x3; x4.set_label("x4");

	x4.set_grad(1.0f);
	x4.backward();
	trace(x4);

	std::cout << "result: " << x4 << std::endl;


	EXPECT_NEAR(x4, 3.386f, 0.01);
}