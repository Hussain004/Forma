declare module 'dagre' {
  export interface GraphNode {
    x: number;
    y: number;
    width: number;
    height: number;
  }

  export class Graph {
    setGraph(opts: Record<string, unknown>): this;
    setDefaultEdgeLabel(fn: () => Record<string, unknown>): this;
    setNode(id: string, opts: { width: number; height: number }): this;
    setEdge(source: string, target: string): this;
    node(id: string): GraphNode;
  }

  export const graphlib: { Graph: typeof Graph };
  export function layout(g: Graph): void;

  const dagre: {
    graphlib: { Graph: typeof Graph };
    layout: typeof layout;
  };
  export default dagre;
}
