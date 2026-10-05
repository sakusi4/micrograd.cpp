#include <print>
#include <memory>
#include <vector>
#include <unordered_set>
#include <functional>
#include <algorithm>
#include <random>

class Value {
public:
    struct Node {
        double value;
        double grad{0.0};
        std::vector<std::shared_ptr<Node>> prev;
        std::function<void()> _backward;

        Node(double v, std::vector<std::shared_ptr<Node>> children = {})
            : value(v), prev(std::move(children)) {}
    };

    std::shared_ptr<Node> node;

    Value(double _value, std::vector<std::shared_ptr<Node>> children = {})
        : node(std::make_shared<Node>(_value, std::move(children))) {
    }

    Value operator+(const Value& other) const {
        Value ret = Value(this->node->value + other.node->value, {this->node, other.node});

        ret.node->_backward = [ret_node = ret.node.get(), this_node = this->node, other_node = other.node]() {
            this_node->grad += ret_node->grad;
            other_node->grad += ret_node->grad;
        };

        return ret;
    }

    Value operator-(const Value& other) const {
        Value ret = Value(this->node->value - other.node->value, {this->node, other.node});

        ret.node->_backward = [ret_node = ret.node.get(), this_node = this->node, other_node = other.node]() {
            this_node->grad += ret_node->grad;
            other_node->grad -= ret_node->grad;
        };

        return ret;
    }

    Value operator*(const Value& other) const {
        Value ret = Value(this->node->value * other.node->value, {this->node, other.node});

        ret.node->_backward = [ret_node = ret.node.get(), this_node = this->node, other_node = other.node]() {
            this_node->grad += other_node->value * ret_node->grad;
            other_node->grad += this_node->value * ret_node->grad;
        };

        return ret;
    }

    Value relu() const {
        Value ret = Value(std::max(0.0, this->node->value), {this->node});

        ret.node->_backward = [ret_node = ret.node.get(), this_node = this->node]() {
            if (this_node->value > 0) {
                this_node->grad += ret_node->grad;
            }
        };

        return ret;
    }

    double getValue() const {
        return node->value;
    }

    void backward() {
        std::vector<Node*> topo;
        std::unordered_set<Node*> visited;

        auto build_topo = [&](auto& self, Node* n) -> void {
            if (!n || visited.contains(n))
                return;

            visited.insert(n);
            for (const auto& child : n->prev) {
                self(self, child.get());
            }

            topo.push_back(n);
        };

        build_topo(build_topo, this->node.get());

        this->node->grad = 1.0;
        for (auto it = topo.rbegin(); it != topo.rend(); it++) {
            if ((*it)->_backward) {
                (*it)->_backward();
            }
        }
    }
};

class Neuron {
public:
    bool nonlin{true}; // true for hidden layers, false for output layer

    std::vector<Value> w;
    Value b{0};

    Neuron(int nin, bool nonlin = true) : nonlin(nonlin) {
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (int i = 0; i < nin; i++) {
            w.push_back(Value(dist(gen), {}));
        }

        b.node->value = 0.0;
    }

    Value forward(const std::vector<Value>& inputs) {
        Value sum = b; // prevent making extra dummy node
        for(size_t i = 0; i < inputs.size(); i++) {
            sum = sum + (this->w[i] * inputs[i]);
        }

        return nonlin ? sum.relu() : sum;
    }

    std::vector<Value*> parameters() {
        std::vector<Value*> ret;
        for (auto& itr : w) {
            ret.push_back(&itr);
        }

        ret.push_back(&b);

        return ret;
    }
};

class Layer {
public:
    std::vector<Neuron> n;

    Layer(int nin, int nout, bool nonlin = true) {
        for (int i = 0; i < nout; i++) {
            n.push_back(Neuron(nin, nonlin));
        }
    }

    std::vector<Value> forward(const std::vector<Value>& inputs) {
        std::vector<Value> out;
        for(auto& itr : n) {
            out.push_back(itr.forward(inputs));
        }

        return out;
    }

    std::vector<Value*> parameters() {
        std::vector<Value*> ret;
        for (auto& itr : n) {
            for (auto& parameter : itr.parameters()) {
                ret.push_back(parameter);
            }
        }

        return ret;
    }
};

class MLP {
public:
    std::vector<Layer> l;

    MLP(int nin, const std::vector<int>& nouts) {
        std::vector<int> sz = {nin};
        sz.insert(sz.end(), nouts.begin(), nouts.end());

        for (size_t i = 0; i < nouts.size(); i++) {
            bool nonlin = (i != nouts.size() - 1);
            l.push_back(Layer(sz[i], sz[i + 1], nonlin));
        }
    }

    std::vector<Value> forward(std::vector<Value> inputs) {
        for (size_t i = 0; i < l.size(); i++) {
            inputs = l[i].forward(inputs);
        }

        return inputs;
    }

    std::vector<Value*> parameters() {
        std::vector<Value*> ret;
        for (auto& itr : l) {
            for (auto& parameter : itr.parameters()) {
                ret.push_back(parameter);
            }
        }

        return ret;
    }
};

int main() {
    MLP model(3, {4, 4, 1});

    std::vector<std::vector<Value>> xs = {
        {Value(2.0), Value(3.0), Value(-1.0)},
        {Value(3.0), Value(-1.0), Value(0.5)},
        {Value(0.5), Value(1.0), Value(1.0)},
        {Value(1.0), Value(1.0), Value(-1.0)}
    };

    std::vector<double> ys = { 1.0, -1.0, -1.0, 1.0 };

    for (int epoch = 0; epoch < 50; epoch++) {
        Value total_loss = 0;

        for (size_t i = 0; i < xs.size(); i++) {
            Value y_pred = model.forward(xs[i])[0];
            Value diff = y_pred - Value(ys[i]);
            Value sqerr = diff * diff;

            total_loss = total_loss + sqerr;
        }

        for (auto* p : model.parameters()) {
            p->node->grad = 0.0;
        }

        total_loss.backward();

        for (auto* p : model.parameters()) {
            p->node->value -= 0.02 * p->node->grad;
        }

        std::println("Epoch {}: loss = {:.4f}", epoch, total_loss.getValue());
    }

    for (size_t i = 0; i < xs.size(); i++) {
        Value y_pred = model.forward(xs[i])[0];
        std::println("y_pred => {}, ys => {}", y_pred.getValue(), ys[i]);
    }

    return 0;
}
