import { Matrix } from 'vectorious';
import { randomString, NodeAndValue, Value } from './util';

export class BaseNode {
  inboundNodes: BaseNode[] = [];
  outboundNodes: BaseNode[] = [];
  readonly id: string = randomString();
  value: Matrix = null;
  gradients: { [key: string]: Matrix } = {};

  constructor(inboundNodes: BaseNode[] = []) {
    this.inboundNodes = inboundNodes;

    this.inboundNodes.forEach(n => {
      n.outboundNodes.push(this);
    });
  }

  toString(): string {
    if (this.value) {
      if (this.value.shape[0] == 1 && this.value.shape[1] == 1) {
        return this.value.get(0, 0).toString();
      }

      return this.value.toString();
    }

    return '';
  }

  forward(): void {
    throw 'Not Implemented';
  }

  backward(): void {
    throw 'Not Implemented';
  }

  protected initializeGradients() {
    this.inboundNodes.forEach(n => {
      this.gradients[n.id] = n.value.map(_ => 0);
    });
  }
}

