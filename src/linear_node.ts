import { Matrix } from 'vectorious';
import { BaseNode } from './node';
import { processMatrices, addMatrixColumns } from './util';

export class LinearNode extends BaseNode {
  constructor(inputs: BaseNode, weights: BaseNode, bias: BaseNode) {
    super([inputs, weights, bias]);
  }

  forward(): void {
    const inputs = this.inboundNodes[0].value;
    const weights = this.inboundNodes[1].value;
    const bias = this.inboundNodes[2].value;

    try {
      this.value = processMatrices('Addition', Matrix.multiply(inputs, weights), bias);
    } catch (e) {
      console.log(`Inputs (${inputs.shape}): 
${inputs} 

Weights (${weights.shape}): 
${weights}

Bias (${bias.shape}): 
${bias}

Result (${this.value.shape}):
${this.value}`);

      throw e;
    }
  }

  backward(): void {
    this.initializeGradients();

    const inputs = this.inboundNodes[0];
    const weights = this.inboundNodes[1];
    const bias = this.inboundNodes[2];

    this.outboundNodes.forEach(n => {
      const gradCost = n.gradients[this.id];

      // console.log(weights.value, gradCost);
      this.gradients[inputs.id] = processMatrices('Addition',
        this.gradients[inputs.id], Matrix.multiply(gradCost, weights.value.T));

      this.gradients[weights.id] = processMatrices('Addition',
        this.gradients[weights.id], Matrix.multiply(inputs.value.T, gradCost));

      this.gradients[bias.id] = processMatrices('Addition',
        this.gradients[bias.id], addMatrixColumns(gradCost));
    });
  }
}

export function Linear(inputs: BaseNode, weights: BaseNode, bias: BaseNode): LinearNode {
  return new LinearNode(inputs, weights, bias);
}