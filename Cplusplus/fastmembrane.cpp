#include <cstdio>
#include<iostream>
#include<math.h>
#include <random>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <iomanip>
/* PARTICLE */

class Particle{

public:
  float posx, posy, posz;
  float velx, vely, velz;
  float spring_forcex, spring_forcey, spring_forcez;
  float accx, accy, accz;

  void update_acceleration(float extfx, float extfy, float extfz, float feedx, float feedy, float feedz);
  void update_velocity();
  void update_position();
  void leapfrog();

  float w_in;
  bool fixed;
  float w_feed;

  void set_values(std::string posx_in, std::string posy_in, std::string posz_in, std::string mass_in, std::string fixed_in,
  std::string w_feed_in, std::string inputcoef_in, float dt_in, int net_dim_in);

  float mass;
  float velhalfx, velhalfy, velhalfz;
  float prioraccx, prioraccy, prioraccz;
  float gravityx;
  float gravityy;
  float gravityz;
  float dt;
  float net_dim;

};

void Particle::set_values(std::string posx_in, std::string posy_in, std::string posz_in, std::string mass_in, std::string fixed_in,
std::string w_feed_in, std::string inputcoef_in, float dt_in, int net_dim_in)
{
  posx = std::stof(posx_in);
  posy = std::stof(posy_in);
  posz = std::stof(posz_in);

  velx = 0.0;
  vely = 0.0;
  velz = 0.0;
  accx = 0.0;
  accy = 0.0;
  accz = 0.0;

  mass = std::stof(mass_in);
  spring_forcex = 0.0;
  spring_forcey = 0.0;
  spring_forcez = 0.0;
  dt = dt_in;
  net_dim = net_dim_in;

  velhalfx = 0.0;
  velhalfy = 0.0;
  velhalfz = 0.0;
  prioraccx = 0.0;
  prioraccy = 0.0;
  prioraccz =0.0;
  w_feed= std::stof(w_feed_in);
  w_in = std::stof(inputcoef_in);

  if(fixed_in=="True"){
    fixed = true;
  }
  else{
    fixed = false;
  }

  gravityx = 0.0;
  gravityy = 0.0;
  gravityz = 0.0;

}

void Particle::update_acceleration(float extfx, float extfy, float extfz, float feedx, float feedy, float feedz)
{

  if (net_dim==2){
      prioraccx = accx;
      prioraccy = accy;
      if (!fixed){
        accx = (spring_forcex + extfx*w_in + feedx*w_feed + gravityx)/mass;
        accy = (spring_forcey + extfy*w_in + feedy*w_feed + gravityy)/mass;
        spring_forcex = 0.0;
        spring_forcey = 0.0;
        }
  }
  if (net_dim==3){
      prioraccx = accx;
      prioraccy = accy;
      prioraccz = accz;
      if (!fixed){
        accx = (spring_forcex + extfx*w_in + feedx*w_feed + gravityx)/mass;
        accy = (spring_forcey + extfy*w_in + feedy*w_feed + gravityy)/mass;
        accz = (spring_forcez + extfz*w_in + feedz*w_feed + gravityz)/mass;
        spring_forcex = 0.0;
        spring_forcey = 0.0;
        spring_forcez = 0.0;
        }
  }


}

void Particle::update_velocity()
{
  if (net_dim==2){
   velx += accx*dt;
   vely += accy*dt;
  }
  if (net_dim==3){
   velx += accx*dt;
   vely += accy*dt;
   velz += accz*dt;
  }

}

void Particle::update_position()
{
  if(net_dim==2){
    posx += velx*dt;
    posy += vely*dt;
  }
  if (net_dim==3){
    posx += velx*dt;
    posy += vely*dt;
    posz += velz*dt;
  }

}

void Particle::leapfrog()
{
  if(net_dim==2){
  velhalfx = velx + prioraccx*dt/2.0;
  velhalfy = vely + prioraccy*dt/2.0;

  posx = posx + velhalfx*dt;
  posy = posy + velhalfy*dt;

  velx = velhalfx + accx*dt/2.0;
  vely = velhalfy + accy*dt/2.0;
  }

  if(net_dim==3){
  velhalfx = velx + prioraccx*dt/2.0;
  velhalfy = vely + prioraccy*dt/2.0;
  velhalfz = velz + prioraccz*dt/2.0;

  posx = posx + velhalfx*dt;
  posy = posy + velhalfy*dt;
  posz = posz + velhalfz*dt;

  velx = velhalfx + accx*dt/2.0;
  vely = velhalfy + accy*dt/2.0;
  velz = velhalfz + accz*dt/2.0;
  }
}

/* SPRING */

class Spring
{

public:
  void set_values(std::string l0_in, std::string p1_in, std::string p2_in,
std::string k1_in, std::string k2_in, std::string d1_in,
std::string d2_in, int net_dim_in);
  std::vector<Particle> Calulateforce(std::vector<Particle> particles);

  float l0;
  float lt;
  int p1;
  int p2;
  float k1;
  float k2;
  float d1;
  float d2;

  float pos1x;
  float pos1y;
  float pos1z;
  float pos2x ;
  float pos2y ;
  float pos2z;

  float vel1x;
  float vel1y;
  float vel1z;
  float vel2x;
  float vel2y;
  float vel2z;

  float F_spring;
  float F_damperx;
  float F_dampery;
  float F_damperz;
  float net_dim;

};

void Spring::set_values(std::string l0_in, std::string p1_in, std::string p2_in,
std::string k1_in, std::string k2_in, std::string d1_in,
std::string d2_in, int net_dim_in)
{
  l0 = std::stof(l0_in);
  p1 = std::stof(p1_in);
  p2 = std::stof(p2_in);
  k1 = std::stof(k1_in);
  k2 = std::stof(k2_in);
  d1 = std::stof(d1_in);
  d2 = std::stof(d2_in);
  net_dim = net_dim_in;
}

std::vector<Particle> Spring::Calulateforce(std::vector<Particle> particles)
{

  if (net_dim ==2){
      pos1x = particles[p1].posx;
      pos1y = particles[p1].posy;
      pos2x = particles[p2].posx;
      pos2y = particles[p2].posy;

      vel1x = particles[p1].velx;
      vel1y = particles[p1].vely;
      vel2x = particles[p2].velx;
      vel2y = particles[p2].vely;

      lt = sqrt(pow(pos2x-pos1x, 2) + pow(pos2y-pos1y, 2));

      F_spring = k1*pow((lt-l0),3) + k2*(lt-l0);
      F_damperx = d1*pow(((vel2x-vel1x)*(pos2x-pos1x)/lt), 3) + d2*((vel2x-vel1x)*(pos2x-pos1x)/lt);
      F_dampery = d1*pow(((vel2y-vel1y)*(pos2y-pos1y)/lt), 3) + d2*((vel2y-vel1y)*(pos2y-pos1y)/lt);


      particles[p1].spring_forcex += (F_spring + F_damperx)*(pos2x-pos1x)/lt;
      particles[p1].spring_forcey += (F_spring + F_dampery)*(pos2y-pos1y)/lt;
      particles[p2].spring_forcex -= (F_spring + F_damperx)*(pos2x-pos1x)/lt;
      particles[p2].spring_forcey -= (F_spring + F_dampery)*(pos2y-pos1y)/lt;

  }
  if (net_dim ==3){
      pos1x = particles[p1].posx;
      pos1y = particles[p1].posy;
      pos1z = particles[p1].posz;

      pos2x = particles[p2].posx;
      pos2y = particles[p2].posy;
      pos2z = particles[p2].posz;

      vel1x = particles[p1].velx;
      vel1y = particles[p1].vely;
      vel1z = particles[p1].velz;

      vel2x = particles[p2].velx;
      vel2y = particles[p2].vely;
      vel2z = particles[p2].velz;

      lt = sqrt(pow(pos2x-pos1x, 2) + pow(pos2y-pos1y, 2) + pow(pos2z-pos1z, 2));

      F_spring = k1*pow((lt-l0),3) + k2*(lt-l0);
      F_damperx = d1*pow(((vel2x-vel1x)*(pos2x-pos1x)/lt), 3) + d2*((vel2x-vel1x)*(pos2x-pos1x)/lt);
      F_dampery = d1*pow(((vel2y-vel1y)*(pos2y-pos1y)/lt), 3) + d2*((vel2y-vel1y)*(pos2y-pos1y)/lt);
      F_damperz = d1*pow(((vel2z-vel1z)*(pos2z-pos1z)/lt), 3) + d2*((vel2z-vel1z)*(pos2z-pos1z)/lt);


      particles[p1].spring_forcex += (F_spring + F_damperx)*(pos2x-pos1x)/lt;
      particles[p1].spring_forcey += (F_spring + F_dampery)*(pos2y-pos1y)/lt;
      particles[p1].spring_forcez += (F_spring + F_damperz)*(pos2z-pos1z)/lt;

      particles[p2].spring_forcex -= (F_spring + F_damperx)*(pos2x-pos1x)/lt;
      particles[p2].spring_forcey -= (F_spring + F_dampery)*(pos2y-pos1y)/lt;
      particles[p2].spring_forcez -= (F_spring + F_damperz)*(pos2z-pos1z)/lt;

  }

  return particles;
}

/* MEMBRANE */

class Membrane
{

public:
 void set_values(std::vector<Particle> particles_in, std::vector<Spring> springs_in,
                   int nb_edge_in, float washout_time_in, float dt_in, float total_time_in, int net_dim_in);
 void set_attributes(std::string action_in, std::vector<std::vector<float> > M_in,
   std::vector<float> force_in, std::vector<float> y_in, std::vector<float> w_in, std::string integrator_in);
 std::vector<std::vector<float> > run_loopless();
 std::vector<std::vector<float> >  run_openloop();
 std::vector<std::vector<float> >  run_closedloop();

  std::string action;
  std::vector<std::vector<float> > M ;
  std::vector<float> force;
  std::vector<float> y;
  std::vector<float> w;
  std::string integrator;
  std::vector<Spring> springs;
  std::vector<Particle> particles;
  int nb_edge;
  float washout_time;
  float dt;
  float total_time;
  int net_dim;

};

void Membrane::set_values(std::vector<Particle> particles_in, std::vector<Spring> springs_in,
int nb_edge_in, float washout_time_in, float dt_in, float total_time_in, int net_dim_in)
{
  nb_edge = nb_edge_in;
  washout_time = washout_time_in;
  particles = particles_in;
  springs = springs_in;
  dt = dt_in;
  total_time = total_time_in;
  net_dim = net_dim_in;
}


void Membrane::set_attributes(std::string action_in, std::vector<std::vector<float> > M_in,
  std::vector<float> force_in, std::vector<float> y_in, std::vector<float> w_in, std::string integrator_in)
{
  action = action_in;
  M = M_in;
  integrator = integrator_in;

  if(action== "closedloop"){
    force = force_in;
    w = w_in;
  }
  else if (action=="loopless"){
    force=force_in;
  }
  else if (action=="openloop"){
    force=force_in;
    y=y_in;
  }
}

std::vector<std::vector<float> > Membrane::run_loopless()
{
  int i_t=0;

  for (float t=0.0; t<total_time;  t+=dt)
  {

    for (int j=0; j<springs.size(); j++)
    {
      particles = springs[j].Calulateforce(particles);
      M[j][i_t]=springs[j].lt - springs[j].l0;
    }


    for (int p=0; p<particles.size(); p++)
    {
      if (net_dim==2){
        particles[p].update_acceleration(force[i_t], 0.0, 0.0, 0.0, 0.0, 0.0);
      }
      if (net_dim==3){
        particles[p].update_acceleration(0.0, 0.0, -force[i_t], 0.0, 0.0, 0.0);
      }

      if (integrator=="euler") {
        particles[p].update_velocity();
        particles[p].update_position();
      }
      if (integrator=="verlet") {
        particles[p].leapfrog();
      }
    }
    i_t++;
  }

  return M;
}

std::vector<std::vector<float> > Membrane::run_openloop()
{

  int i_t=0;
  int total_part = particles.size();

  for (float t=0.0; t<total_time; t+=dt){

    for (int j=0; j<springs.size(); j++){
      particles = springs[j].Calulateforce(particles);
      M[j][i_t]= springs[j].lt - springs[j].l0 ;
    }

    for (int p=0; p<total_part; p++){

      if(net_dim==2){
        particles[p].update_acceleration(force[i_t], 0.0, 0.0, y[i_t], 0.0, 0.0);
      }
      if(net_dim==3){
        particles[p].update_acceleration(0.0, 0.0, -force[i_t], y[i_t], 0.0, 0.0);
      }

      if (integrator=="euler") {
        particles[p].update_velocity();
        particles[p].update_position();
      }
      if (integrator=="verlet") {
        particles[p].leapfrog();
      }
    }

    i_t++;
  }

  return M;
}

std::vector<std::vector<float> >  Membrane::run_closedloop()
{

  int i_t=0;
  int total_part = particles.size();
  float feed_val;

  for (float t=0.0; t<total_time;  t+=dt){

    for (int j=0; j<springs.size(); j++){
      particles = springs[j].Calulateforce(particles);
      M[j][i_t]= springs[j].lt - springs[j].l0;
    }

    feed_val = w[0];
    for(int f=0; f<(w.size()-1); f++){
      feed_val += w[f+1]*M[f][i_t];
    }

    for (int p=0; p<total_part; p++){

      if(net_dim==2){
         particles[p].update_acceleration(force[i_t], 0.0, 0.0, feed_val, 0.0, 0.0);
      }
      if (net_dim==3){
        particles[p].update_acceleration(0.0, 0.0, -force[i_t], feed_val, 0.0, 0.0);
      }

      if (integrator=="euler") {
        particles[p].update_velocity();
        particles[p].update_position();
      }
      if (integrator=="verlet") {
        particles[p].leapfrog();
      }

    }
    i_t++;
  }
  return M;
}

void export_matrix(std::vector<std::vector<float> > M_out )
{
  for (int i = 0; i < M_out.size(); i++)
    {
      for (int j = 0; j < M_out[0].size(); j++){
        std::cout<< M_out[i][j] << "*";
      }
    }
}

Membrane Construct_membrane(int argc, char* argv[])
{
  int argc_count = 1;
  std::string action = argv[argc_count];
  argc_count++;
  int nb_particle = atof(argv[argc_count]);
  argc_count++;
  int nb_spring = atof(argv[argc_count]);
  argc_count++;
  float washout_time = atof(argv[argc_count]);
  argc_count++;
  std::string integrator = argv[argc_count];
  argc_count++;
  int dim1 = atof(argv[argc_count]);
  argc_count++;
  int dim2 = atof(argv[argc_count]);
  argc_count++;
  float dt = atof(argv[argc_count]);
  argc_count++;
  float total_time = atof(argv[argc_count]);
  argc_count++;
  int net_dim = atof(argv[argc_count]);
  argc_count++;

  std::vector<std::vector<float> > M(dim1,std::vector<float>(dim2));
  std::vector<float> w;
  std::vector<float> force;
  std::vector<float> y;

  std::string row;
  std::string filename;
  if (action == "openloop"){
    filename = argv[argc_count];
    argc_count++;
    std::ifstream in( filename );
    while( getline( in, row ) ){
      force.push_back(std::stof(row));
   }
   in.close();
   filename = argv[argc_count];
   argc_count++;
   std::ifstream in_sec( filename );
   while( getline( in_sec, row ) ){
     y.push_back(std::stof(row));
  }
  in.close();
  }

  else if (action == "loopless"){

    filename = argv[argc_count];
    argc_count++;
    std::ifstream in( filename );
    while( getline( in, row ) ){
      force.push_back(std::stof(row));
   }
   in.close();
  }

  else if (action == "closedloop"){
    filename = argv[argc_count];
    argc_count++;
    std::ifstream in( filename );
    while( getline( in, row ) ){
      force.push_back(std::stof(row));
   }
   in.close();
   filename = argv[argc_count];
   argc_count++;
   std::ifstream in_sec( filename );
   while( getline( in_sec, row ) ){
     w.push_back(std::stof(row));
  }
  in.close();
  }

  std::vector<std::string> particle_posx;
  std::vector<std::string> particle_posy;
  std::vector<std::string> particle_posz;
  std::vector<std::string>  particle_mass;
  std::vector<std::string>  particle_fixed;
  std::vector<std::string>  particle_w_feed;
  std::vector<std::string>  particle_input_coef;
  std::vector<std::string>  spring_l0;
  std::vector<std::string>  spring_p1;
  std::vector<std::string> spring_p2;
  std::vector<std::string>  spring_k1;
  std::vector<std::string>  spring_k2;
  std::vector<std::string>  spring_d1;
  std::vector<std::string>  spring_d2;

  for(int i=argc_count; i<argc; i++){

    if(i< (argc_count + nb_particle)){
      particle_posx.push_back(argv[i]);
    }
    else if(i<(argc_count + 2*nb_particle) and i>=(argc_count + nb_particle)){
      particle_posy.push_back(argv[i]);
    }
    else if(i<(argc_count + 3*nb_particle) and i>=(argc_count + 2*nb_particle)){
      particle_mass.push_back(argv[i]);
    }
    else if(i<(argc_count + 4*nb_particle) and i>=(argc_count + 3*nb_particle)){
      particle_fixed.push_back(argv[i]);
    }
    else if(i<(argc_count + 5*nb_particle) and i>=(argc_count + 4*nb_particle)){
      particle_w_feed.push_back(argv[i]);
    }
    else if(i<(argc_count + 6*nb_particle) and i>=(argc_count + 5*nb_particle)){
      particle_input_coef.push_back(argv[i]);
    }
    else if(i<(argc_count + 6*nb_particle + nb_spring) and i>=(argc_count + 6*nb_particle)){
      spring_l0.push_back(argv[i]);
    }
    else if(i<(argc_count + 6*nb_particle + 2*nb_spring) and i>=(argc_count + 6*nb_particle + nb_spring)){
      spring_p1.push_back(argv[i]);
    }
    else if(i<(argc_count + 6*nb_particle + 3*nb_spring) and i>=(argc_count + 6*nb_particle + 2*nb_spring)){
      spring_p2.push_back(argv[i]);
    }
    else if(i<(argc_count + 6*nb_particle + 4*nb_spring) and i>=(argc_count + 6*nb_particle + 3*nb_spring)){
      spring_k1.push_back(argv[i]);
    }
    else if(i<(argc_count + 6*nb_particle + 5*nb_spring) and i>=(argc_count + 6*nb_particle + 4*nb_spring)){
      spring_k2.push_back(argv[i]);
    }
    else if(i<(argc_count + 6*nb_particle + 6*nb_spring) and i>=(argc_count + 6*nb_particle + 5*nb_spring)){
      spring_d1.push_back(argv[i]);
    }
    else if(i<(argc_count + 6*nb_particle + 7*nb_spring) and i>=(argc_count + 6*nb_particle + 6*nb_spring)){
      spring_d2.push_back(argv[i]);
    }
    else if(i<(argc_count + 7*nb_particle + 7*nb_spring) and i>=(argc_count + 6*nb_particle + 7*nb_spring)){
      particle_posz.push_back(argv[i]);
    }
  }


  std::vector<Particle> particles;
  std::vector<Spring> springs;

  for (int p=0; p<nb_particle; p++){
    Particle part = Particle::Particle();
    if(net_dim==3){
        part.set_values(particle_posx[p], particle_posy[p], particle_posz[p], particle_mass[p], particle_fixed[p], particle_w_feed[p],
            particle_input_coef[p], dt, net_dim);
    }
    else{
        part.set_values(particle_posx[p], particle_posy[p], "0.0", particle_mass[p], particle_fixed[p], particle_w_feed[p],
            particle_input_coef[p], dt, net_dim);
    }
    particles.push_back(part);
  }

  for (int s=0; s<nb_spring; s++ ){
    Spring sprin = Spring::Spring();
    sprin.set_values(spring_l0[s], spring_p1[s], spring_p2[s],
    spring_k1[s], spring_k2[s], spring_d1[s],
    spring_d2[s], net_dim);
    springs.push_back(sprin);
  }

  Membrane Memb = Membrane::Membrane();
  Memb.set_values(particles, springs, nb_spring, washout_time, dt, total_time, net_dim);
  Memb.set_attributes(action, M, force, y, w, integrator);

  return Memb;

}

int main(int argc, char* argv[])
{
  Membrane Memb = Construct_membrane(argc, argv);

  if (Memb.action=="loopless"){
    Memb.run_loopless();
  }
  else if(Memb.action=="closedloop"){
    Memb.run_closedloop();
  }
  else if(Memb.action=="openloop"){
    Memb.run_openloop();
  }

  export_matrix(Memb.M);
  return 0;
}
